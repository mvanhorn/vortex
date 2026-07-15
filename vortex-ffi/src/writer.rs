// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::SinkExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use futures::channel::mpsc::Sender;
use parking_lot::Mutex;
use vortex::array::ArrayRef;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteSummary;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::Task;
use vortex::io::session::RuntimeSessionExt;

use crate::RUNTIME;
use crate::array::vx_array;
use crate::dtype::vx_dtype;
use crate::error::try_or_default;
use crate::error::vx_error;
use crate::session::vx_session;
use crate::string::vx_view;

#[expect(non_camel_case_types)]
/// vx_writer can be used to write vx_arrays into a (possibly remote) file.
pub struct vx_writer {
    sender: Mutex<Option<Sender<VortexResult<ArrayRef>>>>,
    writer: Mutex<Option<Task<VortexResult<WriteSummary>>>>,
    dtype: DType,
}

/// Open a writer for a file at "path". "path" is copied.
///
/// "dtype" is used to validate pushed arrays so they would all have the same
/// schema.
///
/// "concurrent_array_limit" is the limit on the number of arrays that are
/// processed concurrently. This limits RAM used for processing.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_writer_open(
    session: *const vx_session,
    path: vx_view,
    dtype: *const vx_dtype,
    concurrent_array_limit: usize,
    error_out: *mut *mut vx_error,
) -> *mut vx_writer {
    let session = vx_session::as_ref(session).clone();
    try_or_default(error_out, || {
        vortex_ensure!(!path.ptr.is_null());
        let path = unsafe { path.as_str() }?.to_string();

        let file_dtype = vx_dtype::as_ref(dtype);
        let (sender, receiver) = mpsc::channel(concurrent_array_limit);
        let dtype = file_dtype.clone();
        let array_stream = ArrayStreamAdapter::new(dtype.clone(), receiver.into_stream());

        let writer = session.handle().spawn(async move {
            let mut file = async_fs::File::create(path).await?;
            session.write_options().write(&mut file, array_stream).await
        });

        Ok(Box::into_raw(Box::new(vx_writer {
            sender: Mutex::new(Some(sender)),
            writer: Mutex::new(Some(writer)),
            dtype,
        })))
    })
}

/// Push an array into a writer. Does not take ownership of array.
///
/// Array ordering across concurrent calls to this function is
/// non-deterministic: vx_writer_push(array1) called concurrently with
/// vx_writer_push(array2) may write array2 first.
///
/// Errors if array's dtype and writer's initialized dtype are different.
/// Errors if writer has already been closed.
///
/// Thread safe.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_writer_push(
    writer: *mut vx_writer,
    array: *const vx_array,
    error_out: *mut *mut vx_error,
) {
    try_or_default(error_out, || {
        vortex_ensure!(!array.is_null());
        vortex_ensure!(!writer.is_null());

        let array = vx_array::as_ref(array);
        let writer = unsafe { &*writer };

        vortex_ensure!(
            *array.dtype() == writer.dtype,
            "array dtype {} does not match writer dtype {}",
            array.dtype(),
            writer.dtype
        );

        let mut sender = writer
            .sender
            .lock()
            .clone()
            .ok_or_else(|| vortex_err!("writer is closed"))?;

        RUNTIME
            .block_on(sender.send(Ok(array.clone())))
            .map_err(|e| vortex_err!("Writer already closed: {e}"))
    })
}

/// Close a writer.
///
/// Call to ensure all values pushed to the writer are indeed written. This
/// call writes the footer to the file. If you don't call this function, file
/// will be left corrupted.
///
/// If this function is called concurrently with vx_writer_push, it will block
/// until vx_writer_push call finishes.
///
/// Thread-unsafe.
///
/// Errors if writer was already closed.
///
/// Use vx_writer_free to free the writer afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_writer_close(
    writer: *mut vx_writer,
    error_out: *mut *mut vx_error,
) {
    try_or_default(error_out, || {
        vortex_ensure!(!writer.is_null());
        let writer = unsafe { &*writer };

        drop(writer.sender.lock().take());

        let writer = writer
            .writer
            .lock()
            .take()
            .ok_or_else(|| vortex_err!("writer is closed"))?;

        RUNTIME.block_on(async {
            let _footer = writer.await?;
            VortexResult::Ok(())
        })
    })
}

/// Release the writer.
///
/// Thread unsafe. Must be called exactly once.
///
/// If vx_writer_close wasn't called before this function, file is left
/// corrupted.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_writer_free(writer: *mut vx_writer) {
    if writer.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(writer) });
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempfile::NamedTempFile;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::buffer;
    use vortex::dtype::DType;

    use super::*;
    use crate::array::vx_array;
    use crate::array::vx_array_free;
    use crate::data_source::vx_data_source_new;
    use crate::data_source::vx_data_source_options;
    use crate::dtype::vx_dtype;
    use crate::dtype::vx_dtype_free;
    use crate::error::vx_error_free;
    use crate::session::vx_session_free;
    use crate::session::vx_session_new;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_writer_basic() {
        unsafe {
            let session = vx_session_new();

            let temp_file = NamedTempFile::new().unwrap();
            let path = vx_view::from_str(temp_file.path().to_str().unwrap());

            let dtype = DType::Primitive(vortex::dtype::PType::I32, false.into());
            let vx_dtype_ptr = vx_dtype::new(Arc::new(dtype));

            let mut error = std::ptr::null_mut();
            let writer = vx_writer_open(session, path, vx_dtype_ptr, &raw mut error);
            assert!(error.is_null());
            assert!(!writer.is_null());

            let array = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);
            let vx_array_ptr = vx_array::new(Arc::new(array.into_array()));

            vx_writer_push(writer, vx_array_ptr, &raw mut error);
            assert!(error.is_null());

            vx_writer_close(writer, &raw mut error);
            assert!(error.is_null());

            vx_writer_free(writer);
            vx_array_free(vx_array_ptr);
            vx_dtype_free(vx_dtype_ptr);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_writer_multiple_arrays() {
        unsafe {
            let session = vx_session_new();

            let temp_file = NamedTempFile::new().unwrap();
            let path = vx_view::from_str(temp_file.path().to_str().unwrap());

            let dtype = DType::Primitive(vortex::dtype::PType::U64, false.into());
            let vx_dtype_ptr = vx_dtype::new(Arc::new(dtype));

            let mut error = std::ptr::null_mut();
            let writer = vx_writer_open(session, path, vx_dtype_ptr, &raw mut error);
            assert!(error.is_null());

            for i in 0..3 {
                let start = i * 3;
                let array = PrimitiveArray::new(
                    buffer![start as u64, (start + 1) as u64, (start + 2) as u64],
                    Validity::NonNullable,
                );
                let vx_array_ptr = vx_array::new(Arc::new(array.into_array()));

                vx_writer_push(writer, vx_array_ptr, &raw mut error);
                assert!(error.is_null());

                vx_array_free(vx_array_ptr);
            }

            vx_writer_close(writer, &raw mut error);
            assert!(error.is_null());
            vx_writer_free(writer);

            vx_dtype_free(vx_dtype_ptr);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_writer_invalid_path() {
        unsafe {
            let session = vx_session_new();

            // Use a path that will fail during file creation (read-only directory on most systems)
            let invalid_path = vx_view::from_str("/dev/null/invalid.vortex");
            let dtype = DType::Primitive(vortex::dtype::PType::I32, false.into());
            let vx_dtype_ptr = vx_dtype::new(Arc::new(dtype));

            let mut error = std::ptr::null_mut();
            let writer = vx_writer_open(session, invalid_path, vx_dtype_ptr, &raw mut error);

            // Creation may succeed but close should fail due to invalid path
            if !writer.is_null() {
                let array = PrimitiveArray::new(buffer![1i32], Validity::NonNullable);
                let vx_array_ptr = vx_array::new(Arc::new(array.into_array()));
                vx_writer_push(writer, vx_array_ptr, &raw mut error);
                vx_array_free(vx_array_ptr);

                vx_writer_close(writer, &raw mut error);
                // Either error is set or operation succeeds (depends on filesystem)
                if !error.is_null() {
                    vx_error_free(error);
                }
                vx_writer_free(writer);
            } else if !error.is_null() {
                vx_error_free(error);
            }

            vx_dtype_free(vx_dtype_ptr);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_writer_free() {
        let dtype = DType::Primitive(vortex::dtype::PType::I32, false.into());
        let array = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);

        let file = NamedTempFile::new().unwrap();
        let path = vx_view::from_str(file.path().to_str().unwrap());

        unsafe {
            let session = vx_session_new();
            let dtype = vx_dtype::new(Arc::new(dtype));

            let mut error = std::ptr::null_mut();
            let writer = vx_writer_open(session, path, dtype, &raw mut error);
            assert!(error.is_null());

            let array = vx_array::new(Arc::new(array.into_array()));
            vx_writer_push(writer, array, &raw mut error);
            assert!(error.is_null());
            vx_array_free(array);

            vx_writer_free(writer);

            let opts = vx_data_source_options {
                paths: &raw const path,
                paths_len: 1,
            };
            let ds = vx_data_source_new(session, &raw const opts, &raw mut error);
            assert!(ds.is_null());
            assert!(!error.is_null());
            vx_error_free(error);

            vx_dtype_free(dtype);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_writer_null_path() {
        unsafe {
            let session = vx_session_new();

            let dtype = DType::Primitive(vortex::dtype::PType::I32, false.into());
            let vx_dtype_ptr = vx_dtype::new(Arc::new(dtype));

            let mut error = std::ptr::null_mut();
            let writer = vx_writer_open(session, vx_view::null(), vx_dtype_ptr, &raw mut error);

            assert!(writer.is_null());
            assert!(!error.is_null());

            vx_error_free(error);
            vx_dtype_free(vx_dtype_ptr);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_writer_push_after_close() {
        unsafe {
            let session = vx_session_new();

            let temp_file = NamedTempFile::new().unwrap();
            let path = vx_view::from_str(temp_file.path().to_str().unwrap());

            let dtype = DType::Primitive(vortex::dtype::PType::I32, false.into());
            let vx_dtype_ptr = vx_dtype::new(Arc::new(dtype));

            let mut error = std::ptr::null_mut();
            let writer = vx_writer_open(session, path, vx_dtype_ptr, &raw mut error);
            assert!(error.is_null());

            let array = PrimitiveArray::new(buffer![1i32], Validity::NonNullable);
            let vx_array_ptr = vx_array::new(Arc::new(array.into_array()));

            vx_writer_push(writer, vx_array_ptr, &raw mut error);
            assert!(error.is_null());

            vx_writer_close(writer, &raw mut error);
            assert!(error.is_null());

            vx_writer_push(writer, vx_array_ptr, &raw mut error);
            assert!(!error.is_null());
            let message = crate::error::vx_error_message(error);
            assert!(message.as_str().unwrap().contains("closed"));
            vx_error_free(error);

            vx_writer_close(writer, &raw mut error);
            assert!(!error.is_null());
            vx_error_free(error);

            vx_writer_free(writer);

            vx_array_free(vx_array_ptr);
            vx_dtype_free(vx_dtype_ptr);
            vx_session_free(session);
        }
    }
}

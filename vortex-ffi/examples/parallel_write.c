// SPDX-License-Identifier: CC-BY-4.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "vortex.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MAX_THREADS 64
#define CHUNK_LEN 300000

const char *usage = "Write a .vortex file from multiple threads concurrently\n"
                    "Usage: parallel_write [-j threads] [-n chunks-per-thread] <output file path>\n";

void print_error(const char *what, const vx_error *error) {
    const vx_view str = vx_error_message(error);
    fprintf(stderr, "%s: %.*s\n", what, (int)str.len, str.ptr);
}

const vx_array *chunk_array(uint64_t start, uint64_t len) {
    uint64_t *data = malloc(len * sizeof(uint64_t));
    for (uint64_t i = 0; i < len; ++i) {
        data[i] = (start + i) % 997;
    }

    vx_validity validity = {.type = VX_VALIDITY_NON_NULLABLE};
    vx_error *error = NULL;
    const vx_array *array = vx_array_new_primitive(PTYPE_U64, data, len, &validity, &error);
    free(data);
    if (error != NULL) {
        print_error("Error creating chunk array", error);
        vx_error_free(error);
        return NULL;
    }
    return array;
}

struct worker {
    pthread_t thread_id;
    vx_writer *writer;
    uint64_t chunks_per_thread;
    vx_error *error;
};

void *write_thread(void *arg) {
    struct worker *worker = arg;

    for (uint64_t chunk = 0; chunk < worker->chunks_per_thread; ++chunk) {
        const uint64_t index = worker->thread_id * worker->chunks_per_thread + chunk;
        const vx_array *array = chunk_array(index * CHUNK_LEN, CHUNK_LEN);
        if (array == NULL) {
            return NULL;
        }

        vx_writer_push(worker->writer, array, &worker->error);
        vx_array_free(array);
        if (worker->error != NULL) {
            return NULL;
        }
    }

    printf("Thread %lu finished, pushed %lu chunks\n", worker->thread_id + 1, worker->chunks_per_thread);
    return NULL;
}

vx_error *parallel_write(vx_writer *writer, uint64_t num_threads, uint64_t chunks_per_thread) {
    pthread_t threads[MAX_THREADS];
    struct worker infos[MAX_THREADS] = {0};

    printf("Writing using %lu threads, %lu chunks per thread\n", num_threads, chunks_per_thread);
    for (uint64_t id = 0; id < num_threads; ++id) {
        struct worker *info = &infos[id];
        info->thread_id = id;
        info->writer = writer;
        info->chunks_per_thread = chunks_per_thread;
        pthread_create(&threads[id], NULL, write_thread, info);
    }

    for (uint64_t id = 0; id < num_threads; ++id) {
        pthread_join(threads[id], NULL);
    }

    for (uint64_t id = 0; id < num_threads; ++id) {
        if (infos[id].error != NULL) {
            // Don't return other threads' errors, only the first one found
            return infos[id].error;
        }
    }

    return NULL;
}

int parse_options(int argc, char *argv[], uint64_t *threads, uint64_t *chunks_per_thread, char **output) {
    int opt;
    while ((opt = getopt(argc, argv, "j:n:")) != -1) {
        switch (opt) {
        case 'j':
            *threads = strtoul(optarg, NULL, 10);
            break;
        case 'n':
            *chunks_per_thread = strtoul(optarg, NULL, 10);
            break;
        default:
            fprintf(stderr, "%s", usage);
            return 1;
        }
    }

    if (*threads < 1 || *threads > MAX_THREADS) {
        fprintf(stderr, "Invalid thread count %lu, expected [1; %d]\n", *threads, MAX_THREADS);
        return 1;
    }

    if (optind + 1 != argc) {
        fprintf(stderr, "%s", usage);
        return 1;
    }

    *output = argv[optind];
    return 0;
}

int main(int argc, char *argv[]) {
    uint64_t threads = 4;
    uint64_t chunks_per_thread = 3;
    char *output;
    if (parse_options(argc, argv, &threads, &chunks_per_thread, &output)) {
        return 1;
    }

    vx_session *session = vx_session_new();
    if (session == NULL) {
        fprintf(stderr, "Failed to create Vortex session\n");
        return 1;
    }

    const vx_dtype *dtype = vx_dtype_new_primitive(PTYPE_U64, false);

    vx_error *error = NULL;
    vx_writer *writer = vx_writer_open(session, vx_view_from_cstr(output), dtype, threads, &error);
    vx_dtype_free(dtype);
    if (writer == NULL) {
        print_error("Failed to open writer", error);
        vx_error_free(error);
        vx_session_free(session);
        return 1;
    }

    // vx_writer supports being pushed to concurrently from multiple threads.
    error = parallel_write(writer, threads, chunks_per_thread);
    if (error != NULL) {
        print_error("Failed to write", error);
        vx_error_free(error);
        vx_writer_close(writer, &error);
        vx_error_free(error);
        vx_writer_free(writer);
        vx_session_free(session);
        return 1;
    }

    vx_writer_close(writer, &error);
    if (error != NULL) {
        print_error("Error closing writer", error);
        vx_error_free(error);
    }
    vx_writer_free(writer);

    printf("Wrote %lu rows to %s\n", threads * chunks_per_thread * CHUNK_LEN, output);

    vx_session_free(session);
    return 0;
}

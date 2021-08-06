import os

import numpy as np
from tqdm import tqdm
from datetime import datetime


def count_pairwise_assignments(binary_matrix):
    n_persons, n_tasks = binary_matrix.shape
    n_pairwise_assignments = {}
    n_pairwise_assignments_list = []

    for person_1 in range(n_persons):
        for person_2 in range(person_1 + 1, n_persons):
            pairwise_tasks = [k for k in range(n_tasks) if binary_matrix[person_1, k] and binary_matrix[person_2, k]]
            n_pairwise_assignments[(person_1, person_2)] = len(pairwise_tasks)
            n_pairwise_assignments_list.append(len(pairwise_tasks))

    avg_pairwise_assignments = np.mean(n_pairwise_assignments_list)
    return n_pairwise_assignments, avg_pairwise_assignments


def save_matrix(binary_matrix, unique_values, unique_counts, max_3_count, prefix, init_id, generation_id):
    unique_counts_str = '__'.join('{}-{}'.format(val, cnt) for val, cnt in zip(unique_values, unique_counts))
    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out_fname = '{}_init_{}_gen_{}___{}___{}.csv'.format(
        prefix, init_id, generation_id, unique_counts_str, time_stamp)

    print('max_3_count:  {}'.format(max_3_count))
    # print('unique_values:  {}'.format(unique_values))
    # print('unique_counts:  {}'.format(unique_counts))
    print('out_fname:  {}'.format(out_fname))
    np.savetxt(out_fname, binary_matrix, fmt='%d', delimiter='\t')


def get_metrics(binary_matrix):
    n_pairwise_assignments, avg_pairwise_assignments = count_pairwise_assignments(binary_matrix)

    n_pairwise_assignments_list = list(n_pairwise_assignments.values())
    unique_values, unique_counts = np.unique(n_pairwise_assignments_list, return_counts=True)

    unique_values = list(unique_values)
    unique_counts = list(unique_counts)

    # curr_0_count = 0
    # if 0 in unique_values:
    #     curr_0_count = unique_counts[unique_values.index(0)]
    #
    # curr_1_count = 0
    # if 1 in unique_values:
    #     curr_1_count = unique_counts[unique_values.index(1)]
    #
    # curr_2_count = 0
    # if 2 in unique_values:
    #     curr_2_count = unique_counts[unique_values.index(2)]

    curr_3_count = 0
    if 3 in unique_values:
        curr_3_count = unique_counts[unique_values.index(3)]

    # curr_lt3_count = curr_0_count + curr_1_count + curr_2_count + curr_3_count
    # print('\nfound new valid assignment {} in {} trials with {} valid/trial'.format(
    #     n_valid_found, trials_id + 1, valid_per_trial))

    # if curr_lt3_count > max_lt3_count:
    #     max_lt3_count = curr_lt3_count
    #     prefix = 'max_lt3_count'
    #     save = 1

    return unique_values, unique_counts, curr_3_count


def main():
    n_tasks = 28
    n_persons = 14

    tasks_per_person = 10
    persons_per_task = 5

    n_trials = int(1e7)
    max_gen_trials = int(5e5)

    init_trials = 0
    gen_init = 1

    load_init = ''
    # load_init = 'evo_max_3_count___1-1__2-5__3-76__4-4__5-5___210805_181606.csv'

    if load_init and os.path.isfile(load_init):
        binary_matrix = np.loadtxt(load_init)
        gen_init = 0

    global_max_3_count = 0
    prefix = 'evo_max_3_count'
    init_id = 0
    while True:

        if gen_init:
            init_id += 1
            while True:
                init_trials += 1

                person_to_n_tasks = np.zeros((n_persons,), dtype=np.ubyte)
                available_persons = list(range(n_persons))
                binary_matrix = np.zeros((n_persons, n_tasks), dtype=np.ubyte)

                valid_found = 1

                for task_id in range(n_tasks):
                    available_persons = [k for k in available_persons if person_to_n_tasks[k] < tasks_per_person]
                    n_available_persons = len(available_persons)

                    if n_available_persons < persons_per_task:
                        # print('Ran out of available_persons in task_id: {}'.format(task_id + 1))
                        valid_found = 0
                        break

                    person_idx = np.random.permutation(available_persons)[:persons_per_task]

                    for i, _idx in enumerate(person_idx):
                        person_to_n_tasks[_idx] += 1

                        binary_matrix[_idx, task_id] = 1

                if not valid_found:
                    continue

                # row_sum = np.count_nonzero(binary_matrix, axis=0)
                col_sum = np.count_nonzero(binary_matrix, axis=1)

                is_valid = np.all(col_sum == tasks_per_person)

                if not is_valid:
                    continue
                break

            print('initialization {} completed in {} trials'.format(init_id, init_trials))

        unique_values, unique_counts, max_3_count = get_metrics(binary_matrix)

        if max_3_count > global_max_3_count:
            global_max_3_count = max_3_count
            save_matrix(binary_matrix, unique_values, unique_counts, max_3_count, prefix, init_id, generation_id)

        parent_binary_matrix = binary_matrix.copy()
        col_idx = list(range(n_tasks))
        row_idx = list(range(n_persons))

        child_trials = 0
        generation_trials = 0
        generation_id = 0

        while True:
            generation_trials += 1
            child_trials += 1

            if generation_trials > max_gen_trials > 0:
                gen_init = 1
                break

            x1, x2 = np.random.permutation(col_idx)[:2]
            y1, y2 = np.random.permutation(row_idx)[:2]

            a, b, c, d = parent_binary_matrix[y1, x1], \
                         parent_binary_matrix[y1, x2], \
                         parent_binary_matrix[y2, x1], \
                         parent_binary_matrix[y2, x2]

            if a != d or a == b or b != c:
                continue

            # print('\nvalid child found in {} trials'.format(child_trials))

            child_trials = 0

            binary_matrix = parent_binary_matrix.copy()
            binary_matrix[y1, x1] = 1 - binary_matrix[y1, x1]
            binary_matrix[y2, x1] = 1 - binary_matrix[y2, x1]
            binary_matrix[y1, x2] = 1 - binary_matrix[y1, x2]
            binary_matrix[y2, x2] = 1 - binary_matrix[y2, x2]

            unique_values, unique_counts, curr_3_count = get_metrics(binary_matrix)

            if curr_3_count <= max_3_count:
                continue

            # row_sum = np.count_nonzero(binary_matrix, axis=0)
            # col_sum = np.count_nonzero(binary_matrix, axis=1)
            # is_valid = np.all(col_sum == tasks_per_person) and np.all(row_sum == persons_per_task)
            # assert is_valid, "invalid assignment matrix generated even by performing valid operations"

            generation_id += 1

            print('\n\ninit {} generation {} found in {} trials'.format(init_id, generation_id, generation_trials))
            generation_trials = 0

            max_3_count = curr_3_count
            parent_binary_matrix = binary_matrix.copy()

            if max_3_count > global_max_3_count:
                global_max_3_count = max_3_count
                save_matrix(binary_matrix, unique_values, unique_counts, curr_3_count, prefix, init_id, generation_id)


if __name__ == '__main__':
    main()

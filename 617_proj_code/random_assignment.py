import numpy as np
from tqdm import tqdm


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


def main():
    n_tasks = 28
    n_persons = 14

    tasks_per_person = 10
    persons_per_task = 5

    n_trials = int(1e7)

    min_avg_pairwise_assignments = np.inf

    n_valid_found = 0

    valid_assignments_list = []

    for trials_id in tqdm(range(n_trials)):

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

        if is_valid:

            is_new = all(np.array_equal(k, binary_matrix) for k in valid_assignments_list)
            if not is_new:
                continue

            valid_assignments_list.append(binary_matrix)

            n_valid_found += 1

            valid_per_trial = float(n_valid_found) / (trials_id + 1)
            n_pairwise_assignments, avg_pairwise_assignments = count_pairwise_assignments(binary_matrix)

            n_pairwise_assignments_list = list(n_pairwise_assignments.values())
            unique_values, unique_counts = np.unique(n_pairwise_assignments_list, return_counts=True)

            print('\nn_pairwise_assignments:  {}'.format(n_pairwise_assignments))
            print('n_pairwise_assignments_list:  {}'.format(n_pairwise_assignments_list))
            print('unique_values:  {}'.format(unique_values))
            print('unique_counts:  {}'.format(unique_counts))

            print('\nfound new valid assignment {} in {} trials with {} valid/trial'.format(
                n_valid_found, trials_id + 1, valid_per_trial))

            if avg_pairwise_assignments < min_avg_pairwise_assignments:
                min_avg_pairwise_assignments = avg_pairwise_assignments

                out_fname = 'opt_binary_matrix_{}_{}_{}.csv'.format(
                    n_valid_found, trials_id, int(min_avg_pairwise_assignments))
            else:
                out_fname = 'binary_matrix_{}_{}_{}.csv'.format(
                    n_valid_found, trials_id, int(min_avg_pairwise_assignments))

            np.savetxt(out_fname, binary_matrix, fmt='%d', delimiter='\t')

            print('avg_pairwise_assignments:  {}'.format(avg_pairwise_assignments))
            print('min_avg_pairwise_assignments:  {}'.format(min_avg_pairwise_assignments))

            print()


if __name__ == '__main__':
    main()

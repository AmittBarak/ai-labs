# Define the read_problems_from_file function
def read_problems_from_file(file_path):
    problems = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_problem = None
        for line in lines:
            line = line.strip()
            if line.startswith('u'):
                current_problem = line
                problems[current_problem] = []
            elif current_problem is not None:
                try:
                    item = int(line)
                    problems[current_problem].append(item)
                except ValueError:
                    pass
    return problems


# Define the first_fit algorithm
def first_fit(bin_capacity, items):
    try:
        bins = []
        for item in items:
            placed = False
            for b in bins:
                if b + item <= bin_capacity:
                    bins[bins.index(b)] += item
                    placed = True
                    break
            if not placed:
                bins.append(item)
        return len(bins)
    except Exception as e:
        print(f"An error occurred: {e}")


# Define the function to run the first-fit algorithm from file
def run_first_fit(filename):
    try:
        problems = read_problems_from_file(filename)
        bin_capacity = 150  # Assuming a fixed bin capacity as it is not provided in the file

        results = []
        for problem_id, items in list(problems.items())[:5]:  # Convert dictionary items to a list before slicing
            num_bins = first_fit(bin_capacity, items)
            results.append((problem_id, num_bins))
            print(f"Problem ID: {problem_id}, Number of bins used: {num_bins}")

        return results
    except Exception as e:
        print(f"An error occurred: {e}")
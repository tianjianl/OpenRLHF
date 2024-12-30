import json
import matplotlib.pyplot as plt
import argparse

def read_log_probs(file_path):
    """
    Reads a JSONL file and extracts chosen_log_probs and rejected_log_probs.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        tuple: Two lists containing chosen_log_probs and rejected_log_probs respectively.
    """
    chosen_log_probs = []
    rejected_log_probs = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                try:
                    data = json.loads(line)
                    chosen = data.get('chosen_log_probs')
                    rejected = data.get('rejected_log_probs')
                    
                    # Ensure that the log_probs are present and are numbers
                    if isinstance(chosen, (int, float)):
                        chosen_log_probs.append(chosen)
                    else:
                        print(f"Warning: 'chosen_log_probs' missing or not a number on line {line_number}.")
                    
                    if isinstance(rejected, (int, float)):
                        rejected_log_probs.append(rejected)
                    else:
                        print(f"Warning: 'rejected_log_probs' missing or not a number on line {line_number}.")
                
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {line_number}: {e}")
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        exit(1)
    
    return chosen_log_probs, rejected_log_probs

def plot_histograms(chosen, rejected, bins=50):
    """
    Plots histograms of chosen and rejected log probabilities.

    Args:
        chosen (list): List of chosen_log_probs.
        rejected (list): List of rejected_log_probs.
        bins (int): Number of histogram bins.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot chosen_log_probs
    plt.hist(chosen, bins=bins, color='red', alpha=0.5, label='Chosen Log Probabilities')
    
    # Plot rejected_log_probs
    plt.hist(rejected, bins=bins, color='blue', alpha=0.5, label='Rejected Log Probabilities')
    
    plt.xlabel('Log Probability')
    plt.ylabel('Frequency')
    plt.title('Histogram of Chosen and Rejected Log Probabilities')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lprobs.pdf')
def main():
    parser = argparse.ArgumentParser(description='Plot histograms of chosen and rejected log probabilities from a JSONL file.')
    parser.add_argument('file', help='Path to the JSONL file.')
    parser.add_argument('--bins', type=int, default=50, help='Number of bins for the histogram (default: 50).')
    
    args = parser.parse_args()
    
    chosen_log_probs, rejected_log_probs = read_log_probs(args.file)
    
    if not chosen_log_probs:
        print("No valid 'chosen_log_probs' found. Exiting.")
        exit(1)
    
    if not rejected_log_probs:
        print("No valid 'rejected_log_probs' found. Exiting.")
        exit(1)
    
    plot_histograms(chosen_log_probs, rejected_log_probs, bins=args.bins)

if __name__ == '__main__':
    main()


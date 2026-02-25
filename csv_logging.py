import os
import csv
import time

def csv_logs(run_type, max_iter):
    protection_dir = './protection_results'
    os.makedirs(protection_dir, exist_ok=True)

    stamp = time.strftime('%Y%m%d-%H%M')

    iter_csv_path = os.path.join(protection_dir, f"iter_{run_type}_max{int(max_iter)}_{stamp}.csv")
    summary_csv_path = os.path.join(protection_dir, f"summary_{run_type}_max{int(max_iter)}_{stamp}.csv")

    iter_csv_needs_header = (not os.path.exists(iter_csv_path)) or (os.path.getsize(iter_csv_path) == 0)
    iter_csv_f = open(iter_csv_path, 'a', newline='')
    iter_csv_writer = csv.DictWriter(
        iter_csv_f,
        fieldnames=[
            'run_type',
            'iden',
            'current_iter',
            'max_iter',
            'prior_loss',
            'iden_loss',
            'total_loss',
            'acc',
        ],
    )
    if iter_csv_needs_header:
        iter_csv_writer.writeheader()

    summary_csv_needs_header = (not os.path.exists(summary_csv_path)) or (os.path.getsize(summary_csv_path) == 0)
    summary_csv_f = open(summary_csv_path, 'a', newline='')
    summary_csv_writer = csv.DictWriter(
        summary_csv_f,
        fieldnames=[
            'run_type',
            'max_iter',
            'acc',
            'acc_5',
            'acc_var',
            'acc_var5',
        ],
    )
    if summary_csv_needs_header:
        summary_csv_writer.writeheader()

    return iter_csv_f, iter_csv_writer, summary_csv_f, summary_csv_writer

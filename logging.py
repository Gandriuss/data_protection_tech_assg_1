def csv_logs(run_type, max_iter):

    protection_dir = './protection_results'
    os.makedirs(protection_dir, exist_ok=True)
    iter_csv_path = os.path.join(protection_dir, f"iter_{run_type}.csv")
    summary_csv_path = os.path.join(protection_dir, f"summary_{run_type}.csv")

    iter_csv_f = open(iter_csv_path, 'a', newline='')
    iter_csv_writer = csv.DictWriter(
        iter_csv_f,
        fieldnames=[
            'run_type',
            'iden'
            'current_iter',
            'max_iter',
            'prior_loss',
            'iden_loss',
            'total_loss',
            'acc',
        ],
    )
    iter_csv_writer.writeheader()

    summary_csv_f = open(summary_csv_path, 'a', newline='')
    summary_csv_writer = csv.DictWriter(
        summary_csv_f,
        fieldnames=[
            'run_type',
            'max_iter',
            'aver_acc',
            'aver_acc5',
            'aver_var', 
            'aver_var5'
        ],
    )
    summary_csv_writer.writeheader()

    return iter_csv_f, iter_csv_writer, summary_csv_f, summary_csv_writer


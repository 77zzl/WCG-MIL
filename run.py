import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    
    parser = argparse.ArgumentParser(description='Finite NEST-MIL reproduction queue')
    parser.add_argument('--dataset', choices=['camelyon_plus', 'bracs'], required=True)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--models', nargs='+', choices=['nest-mil', 'nest-mil-core'], default=['nest-mil', 'nest-mil-core'])
    parser.add_argument('--folds', nargs='+', type=int, choices=range(1, 6), default=[1, 2, 3, 4, 5])
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--queue-id', default='paper_reproduction')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.output_dir = str(Path(args.output_dir).resolve())
    args.feature_root = str(Path(args.feature_root).resolve())
    seeds = args.seeds or ([42] if args.dataset == 'camelyon_plus' else [42, 43, 44])
    folds = args.folds if args.dataset == 'camelyon_plus' else [1]
    root = Path(__file__).parent
    queue_dir = Path(args.output_dir).resolve() / 'queues' / args.dataset / args.queue_id
    queue_dir.mkdir(parents=True, exist_ok=True)
    lock = queue_dir / 'runner.lock'
    if lock.exists() and not args.dry_run:
        raise RuntimeError(f'队列锁存在：{lock}。先确认旧进程已退出，再手动删除此锁。')
    jobs = [(model, fold, seed) for model in args.models for fold in folds for seed in seeds]
    if not args.dry_run:
        with lock.open('x', encoding='utf-8') as stream:
            json.dump({'pid': os.getpid(), 'started': datetime.now().isoformat()}, stream)
    try:
        for index, (model, fold, seed) in enumerate(jobs, 1):
            run_id = f'{args.queue_id}_{model}_fold{fold}_seed{seed}'
            directory = Path(args.output_dir) / args.dataset / model / f'fold_{fold}' / f'seed_{seed}' / run_id
            if (directory / 'run_summary.json').exists() and (directory / 'test_result.json').exists() and (directory / 'best_model.pt').exists():
                print(f'跳过已完成 {index}/{len(jobs)}：{run_id}', flush=True)
                continue
            command = [sys.executable, str(root / 'main.py'), '--name', model, '--dataset', args.dataset,
                       '--feature-root', args.feature_root, '--fold', str(fold), '--seed', str(seed),
                       '--output-dir', args.output_dir, '--run-id', run_id, '--efficiency']
            state = directory / 'last_training_state.pt'
            if state.exists():
                command += ['--resume-training', str(state.resolve())]
            print(f'任务 {index}/{len(jobs)}：{run_id}', flush=True)
            if args.dry_run:
                print(subprocess.list2cmdline(command))
                continue
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / 'console.log').open('a', encoding='utf-8') as stream:
                result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'})
            with (queue_dir / 'attempts.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'run_id': run_id, 'returncode': result.returncode,
                                         'ended': datetime.now().isoformat()}) + '\n')
            if result.returncode:
                print(f'失败，继续下一任务；见 {directory / "console.log"}', flush=True)
    finally:
        if not args.dry_run:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

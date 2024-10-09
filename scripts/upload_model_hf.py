import argparse
import os
from huggingface_hub import upload_folder
from rich.console import Console
from rich.panel import Panel
from rich import box, style
from rich. table import Table

CONSOLE = Console(width=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default=None, required=True)
    parser.add_argument("--folder_path", type=str, default=None, required=True)
    parser.add_argument("--token", type=str, default=None, required=False)
    args = parser.parse_args()

    token = args.token or os.getenv("hf_token", None)
    ignore_patterns = ["**/optimizer.bin", "**/random_states*", "**/scaler.pt", "**/scheduler.bin"]

    try:
        if token is not None:
            upload_folder(repo_id=args.repo_id, folder_path=args.folder_path, ignore_patterns=ignore_patterns, token=token)
        else:
            upload_folder(repo_id=args.repo_id, folder_path=args.folder_path, ignore_patterns=ignore_patterns)
        table = Table(title=None, show_header=False, box=box.MINIMAL, title_style=style.Style(bold=True))
        table.add_row(f"Model id {args.repo_id}", str(args.folder_path))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Upload completed DO NOT forget specify the model id in methods! :tada:[/bold]", expand=False))

    except Exception as e:
        CONSOLE.print(f"[bold][yellow]:tada: Upload failed due to {e}.")
        raise e

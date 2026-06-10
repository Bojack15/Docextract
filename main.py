#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from chunker import ChunkConfig
from pipeline import process_file, process_directory, export_json
from vector_store import VectorStore

console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True)
def cli(verbose):
    """DocExtract Ingestion & Vector Storage Pipeline"""
    _setup_logging(verbose)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--chunk-size", default=512, show_default=True)
@click.option("--chunk-overlap", default=50, show_default=True)
@click.option("--ocr-lang", default="eng", show_default=True)
@click.option("--dpi", default=300, show_default=True)
@click.option("--output-json", "-o", default=None, help="Export to JSON")
@click.option("--store/--no-store", default=True, show_default=True)
@click.option("--db-path", default="./vector_db", show_default=True)
def process(path, chunk_size, chunk_overlap, ocr_lang, dpi, output_json, store, db_path):
    """Extract, chunk, and index a document or directory."""
    config = ChunkConfig(size=chunk_size, overlap=chunk_overlap)
    target = Path(path)

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console) as prog:
        task = prog.add_task(f"Processing {target.name}...", total=None)

        if target.is_file():
            docs = [process_file(str(target), config, ocr_lang, dpi)]
        elif target.is_dir():
            docs = process_directory(str(target), config, ocr_lang, dpi)
        else:
            console.print(f"[red]Invalid path: {path}[/red]")
            sys.exit(1)

        prog.update(task, completed=True)

    if not docs:
        console.print("[red]No documents processed.[/red]")
        sys.exit(1)

    for doc in docs:
        _show_summary(doc)

        if output_json:
            stem = Path(doc.filename).stem
            out = output_json if len(docs) == 1 else f"{stem}_extracted.json"
            export_json(doc, out)
            console.print(f"  [green]JSON exported to {out}[/green]")

    if store:
        vs = VectorStore(path=db_path)
        total = sum(vs.add(doc) for doc in docs)
        console.print(f"\n  [green]Indexed {total} total chunks in ChromaDB ({db_path})[/green]")


@cli.command()
@click.argument("query")
@click.option("-n", default=5, show_default=True, help="Number of results")
@click.option("--db-path", default="./vector_db", show_default=True)
@click.option("--filename", default=None, help="Filter by filename")
def search(query, n, db_path, filename):
    """Semantic search against the vector database."""
    vs = VectorStore(path=db_path)
    where = {"filename": filename} if filename else None
    results = vs.search(query, n=n, where=where)

    if not results:
        console.print("[yellow]No matches found.[/yellow]")
        return

    console.print(Panel(f"Search results for: [cyan]{query}[/cyan]", border_style="blue"))

    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        text = r["text"][:500] + "..." if len(r["text"]) > 500 else r["text"]
        dist = r["distance"]
        console.print(f"\n[bold magenta]── Result {i} ──[/bold magenta]")
        console.print(
            f"  Document: [cyan]{meta.get('filename')}[/cyan] · "
            f"Pages: {meta.get('start_page')}-{meta.get('end_page')} · "
            f"Distance: {dist:.4f}"
        )
        console.print(f"  {text}\n")


@cli.command(name="list")
@click.option("--db-path", default="./vector_db", show_default=True)
def list_docs(db_path):
    """List indexed documents in the database."""
    vs = VectorStore(path=db_path)
    docs = vs.list_documents()

    if not docs:
        console.print("[yellow]No documents indexed.[/yellow]")
        return

    table = Table(title="Stored Documents")
    table.add_column("Filename", style="cyan")
    table.add_column("Pages", justify="center")
    table.add_column("Chunks", justify="center")
    table.add_column("Method", style="magenta")
    for d in docs:
        table.add_row(d["filename"], str(d["total_pages"]), str(d["chunks"]), d["method"])
    console.print(table)


@cli.command()
@click.option("--db-path", default="./vector_db", show_default=True)
def stats(db_path):
    """View vector store database statistics."""
    vs = VectorStore(path=db_path)
    info = vs.stats()
    console.print(Panel(
        f"Indexed Chunks: {info['total_chunks']}\n"
        f"Indexed Documents: {info['total_documents']}\n"
        f"Storage Location: {info['path']}",
        title="Vector Store Status", border_style="green",
    ))


def _show_summary(doc):
    table = Table(title=f"{doc.filename}")
    table.add_column("Property", style="cyan", width=18)
    table.add_column("Value")
    table.add_row("Document ID", doc.id[:16] + "...")
    table.add_row("Pages", str(doc.total_pages))
    table.add_row("Words", f"{doc.total_words:,}")
    table.add_row("Chunks", str(len(doc.chunks)))
    table.add_row("Method", doc.method.value)

    for page in doc.pages:
        icon = "[Text]" if page.method.value == "text" else "[OCR]"
        conf = f" ({page.confidence:.1f}%)" if page.confidence is not None else ""
        table.add_row(f"  Page {page.number}", f"{icon} {page.word_count} words{conf}")

    console.print(table)


if __name__ == "__main__":
    cli()



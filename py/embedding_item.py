from re import T
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import polars as pl
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq

# =====================================================
# last-token pooling (Qwen official)
# =====================================================

def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.size(0)
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            seq_lens
        ]

# =====================================================
# resume helper
# =====================================================

def get_written_rows(parquet_path: Path) -> int:
    if not parquet_path.exists():
        return 0
    return (
        pl.scan_parquet(parquet_path)
        .select(pl.count())
        .collect()
        .item()
    )

# =====================================================
# streaming embedding + resume (BATCH progress)
# =====================================================

@torch.no_grad()
def embed_and_write_resume(
    df: pl.DataFrame,
    text_col: str,
    output_path: Path,
    model_name: str,
    max_length: int = 2048,
    batch_size: int = 64,
    target_dim: int = 1024,
    device: str = "cuda",
    do_norm: bool = True,
    emb_scale: float = 1.0
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left"
    )
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    total_rows = len(df)
    written_rows = get_written_rows(output_path)

    if written_rows >= total_rows:
        print("[Resume] All batches already done.")
        return

    remaining_rows = total_rows - written_rows
    num_batches = (remaining_rows + batch_size - 1) // batch_size

    print(
        f"[Resume] total_rows={total_rows}, "
        f"already_done={written_rows}, "
        f"remaining_batches={num_batches}"
    )

    writer = None

    for start in tqdm(
        range(written_rows, total_rows, batch_size),
        total=num_batches,            # ✅ batch 数
        desc="Embedding batches",
        unit="batch",
    ):
        end = min(start + batch_size, total_rows)
        batch_df = df.slice(start, end - start)
        texts = batch_df[text_col].to_list()

        # tokenize
        batch_dict = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        # forward
        outputs = model(**batch_dict)
        emb = last_token_pool(
            outputs.last_hidden_state,
            batch_dict["attention_mask"]
        )
        emb = emb[:, :target_dim]

        if do_norm:
            emb = F.normalize(emb, dim=1)
            emb = emb * emb_scale

        emb_np = emb.cpu().numpy().astype("float32")

        out_df = batch_df.with_columns(
            pl.Series(
                "embedding",
                emb_np.tolist(),
                dtype=pl.List(pl.Float32),
            )
        )

        table = out_df.to_arrow()

        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                table.schema,
                compression="zstd",
            )

        writer.write_table(table)

        # aggressively free
        del emb, emb_np, outputs, batch_dict, out_df, table

    if writer is not None:
        writer.close()

    print(f"[Done] embeddings written to {output_path}")

# =====================================================
# main
# =====================================================

if __name__ == "__main__":

    CATEGORY = "Movies_and_TV"
    DATA_DIR = Path("/home/zihao/llm/llm4rec/data")

    input_path = DATA_DIR / "output" / f"{CATEGORY}_combine_clean.csv"
    output_path = DATA_DIR / "output" / f"{CATEGORY}_combine_clean_embeddings.parquet"

    print(f"[Load] {input_path}")
    df = pl.read_csv(input_path)

    embed_and_write_resume(
        df=df,
        text_col="clean_combine",
        output_path=output_path,
        model_name="/home/zihao/llm/hf_model/Qwen3-Embedding-0.6B",
        batch_size=64,
        max_length=2048,
        target_dim=1024,
        device="cuda" if torch.cuda.is_available() else "cpu",
        do_norm=True,
        emb_scale=1.0
    )

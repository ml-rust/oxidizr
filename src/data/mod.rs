//! Token-stream data loading: row layout, data-parallel ranks, loss masks,
//! the synchronous and prefetching loaders, and the `.tokens.json` sidecar.

mod batch;
mod construction;
mod mask;
mod positioning;
mod prefetch;
mod ranks;
mod rows;
mod source;
mod token_meta;

pub use batch::{Batch, DataLoader};
pub use construction::{create_dummy_data, write_tokens_to_file, LitDataLoader};
pub use mask::TargetMask;
pub use prefetch::{create_prefetch_loader, PrefetchDataLoader};
pub use rows::{DataPosition, RowLayout, RowOrder};
pub use token_meta::{
    read_token_meta, token_meta_path, write_token_meta, InputRecord, TokenMeta, TokenOptions,
    TokenizerRecord, TOKEN_META_SUFFIX, TOKEN_META_VERSION,
};

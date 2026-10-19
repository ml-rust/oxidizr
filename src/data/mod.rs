//! Token-stream data loading: row layout, data-parallel ranks, loss masks,
//! the synchronous and prefetching loaders, data sources, and the
//! `.tokens.json` sidecar.

mod batch;
mod construction;
mod data_config;
mod mask;
mod multi_source;
mod positioning;
mod prefetch;
mod ranks;
mod row_ranges;
mod rows;
mod source;
mod source_record;
mod token_meta;

pub use batch::{Batch, DataLoader};
pub use construction::{create_dummy_data, write_tokens_to_file, LitDataLoader};
pub use data_config::{resolve_data_files, DataConfig, DataSourceSpec};
pub use mask::TargetMask;
pub use prefetch::{create_prefetch_loader, PrefetchDataLoader};
pub use rows::{DataPosition, RowLayout, RowOrder};
pub use source_record::{
    check_meta_sources, set_meta_sources, source_records, DataCheck, DataSourceRecord,
};
pub use token_meta::{
    read_token_meta, token_meta_path, write_token_meta, InputRecord, TokenMeta, TokenOptions,
    TokenizerRecord, TOKEN_META_SUFFIX, TOKEN_META_VERSION,
};

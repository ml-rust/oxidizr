//! `${VAR}` environment-variable expansion for path-shaped config fields.
//!
//! A config must not carry an absolute machine path. A path field instead
//! names an environment variable placeholder — `${BOOSTR_MODELS_DIR}/foo` —
//! and `expand_env_vars` resolves every placeholder at load time, against
//! one mechanism shared by every path field the config has (`checkpoint_dir`
//! today).

use std::env;
use std::fmt;

/// A `${VAR}` placeholder named a variable that is not set in the process
/// environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathExpansionError {
    /// The config field the placeholder came from, e.g. `trainer.checkpoint_dir`.
    pub field: String,
    /// The variable name inside `${...}`.
    pub variable: String,
}

impl fmt::Display for PathExpansionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} references ${{{}}}, but the {} environment variable is not set",
            self.field, self.variable, self.variable
        )
    }
}

impl std::error::Error for PathExpansionError {}

/// Expands every `${VAR}` placeholder in `raw`, reading each variable from
/// the process environment. `field_name` names the config field `raw` came
/// from, for the error message. A string with no `${` is returned unchanged.
///
/// An unterminated `${` (no closing `}`) is left in the output literally
/// rather than treated as an error: it is not the syntax this function
/// expands, so it is not this function's placeholder to reject.
pub fn expand_env_vars(raw: &str, field_name: &str) -> Result<String, PathExpansionError> {
    let mut result = String::with_capacity(raw.len());
    let mut rest = raw;

    while let Some(start) = rest.find("${") {
        let (before, after_marker) = rest.split_at(start);
        result.push_str(before);
        let after_marker = &after_marker[2..]; // skip "${"

        match after_marker.find('}') {
            Some(end) => {
                let variable = &after_marker[..end];
                let value = env::var(variable).map_err(|_| PathExpansionError {
                    field: field_name.to_string(),
                    variable: variable.to_string(),
                })?;
                result.push_str(&value);
                rest = &after_marker[end + 1..];
            }
            None => {
                // No closing brace anywhere in the remainder: keep "${" and
                // everything after it as-is, then stop.
                result.push_str("${");
                result.push_str(after_marker);
                rest = "";
                break;
            }
        }
    }
    result.push_str(rest);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_path_with_no_placeholder_passes_through_unchanged() {
        let out = expand_env_vars("./checkpoints", "trainer.checkpoint_dir").expect("no vars");
        assert_eq!(out, "./checkpoints");
    }

    #[test]
    fn a_single_placeholder_expands_from_the_environment() {
        // SAFETY: test-local env var, no other test reads this name.
        unsafe { env::set_var("OXIDIZR_TEST_PATH_VAR", "/tmp/oxidizr-test-models") };
        let out = expand_env_vars(
            "${OXIDIZR_TEST_PATH_VAR}/g2p_lm/checkpoints/g2p-char",
            "trainer.checkpoint_dir",
        )
        .expect("variable is set");
        assert_eq!(out, "/tmp/oxidizr-test-models/g2p_lm/checkpoints/g2p-char");
        unsafe { env::remove_var("OXIDIZR_TEST_PATH_VAR") };
    }

    #[test]
    fn an_unset_variable_errors_naming_the_field_and_the_variable() {
        unsafe { env::remove_var("OXIDIZR_TEST_MISSING_VAR") };
        let err = expand_env_vars(
            "${OXIDIZR_TEST_MISSING_VAR}/checkpoints",
            "trainer.checkpoint_dir",
        )
        .expect_err("variable is unset");
        assert_eq!(err.field, "trainer.checkpoint_dir");
        assert_eq!(err.variable, "OXIDIZR_TEST_MISSING_VAR");
        assert!(err.to_string().contains("OXIDIZR_TEST_MISSING_VAR"));
    }

    #[test]
    fn multiple_placeholders_in_one_string_all_expand() {
        unsafe {
            env::set_var("OXIDIZR_TEST_A", "aaa");
            env::set_var("OXIDIZR_TEST_B", "bbb");
        }
        let out =
            expand_env_vars("${OXIDIZR_TEST_A}/mid/${OXIDIZR_TEST_B}", "field").expect("both set");
        assert_eq!(out, "aaa/mid/bbb");
        unsafe {
            env::remove_var("OXIDIZR_TEST_A");
            env::remove_var("OXIDIZR_TEST_B");
        }
    }

    #[test]
    fn an_unterminated_placeholder_is_left_literal() {
        let out = expand_env_vars("prefix/${UNCLOSED", "field").expect("no error");
        assert_eq!(out, "prefix/${UNCLOSED");
    }
}

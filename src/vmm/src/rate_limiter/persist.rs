// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Save / restore for the GCRA-backed RateLimiter.
//!
//! Snapshot wire format reflects the GCRA state directly: tat is stored
//! as the deficit at save time (ns until the bucket becomes idle), and
//! reapplied to "now" on restore.

use serde::{Deserialize, Serialize};
use utils::time::TimerFd;

use super::*;
use crate::snapshot::Persist;

/// State for saving a TokenBucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBucketState {
    size: u64,
    one_time_burst: u64,
    initial_one_time_burst: u64,
    refill_time_ms: u64,
    /// Bucket deficit at save time, in ns. `0` = bucket idle/full.
    deficit_ns: u64,
}

impl Persist<'_> for TokenBucket {
    type State = TokenBucketState;
    type ConstructorArgs = ();
    type Error = io::Error;

    fn save(&self) -> Self::State {
        let now = now_ns();
        let tat = self.tat_ns.load(Ordering::Relaxed);
        let deficit_ns = tat.saturating_sub(now);
        TokenBucketState {
            size: self.size,
            one_time_burst: self.one_time_burst.load(Ordering::Relaxed),
            initial_one_time_burst: self.initial_one_time_burst,
            refill_time_ms: self.refill_time_ms,
            deficit_ns,
        }
    }

    fn restore(_: Self::ConstructorArgs, state: &Self::State) -> Result<Self, Self::Error> {
        let bucket = TokenBucket::new(
            state.size,
            state.initial_one_time_burst,
            state.refill_time_ms,
        )
        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        bucket
            .one_time_burst
            .store(state.one_time_burst, Ordering::Relaxed);
        let now = now_ns();
        bucket
            .tat_ns
            .store(now.saturating_add(state.deficit_ns), Ordering::Relaxed);
        Ok(bucket)
    }
}

/// State for saving a RateLimiter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimiterState {
    ops: Option<TokenBucketState>,
    bandwidth: Option<TokenBucketState>,
}

impl Persist<'_> for RateLimiter {
    type State = RateLimiterState;
    type ConstructorArgs = ();
    type Error = io::Error;

    fn save(&self) -> Self::State {
        RateLimiterState {
            ops: self.ops.as_ref().map(|ops| ops.save()),
            bandwidth: self.bandwidth.as_ref().map(|bw| bw.save()),
        }
    }

    fn restore(_: Self::ConstructorArgs, state: &Self::State) -> Result<Self, Self::Error> {
        let rate_limiter = RateLimiter {
            ops: if let Some(ops) = state.ops.as_ref() {
                Some(TokenBucket::restore((), ops)?)
            } else {
                None
            },
            bandwidth: if let Some(bw) = state.bandwidth.as_ref() {
                Some(TokenBucket::restore((), bw)?)
            } else {
                None
            },
            timer_fd: TimerFd::new(),
            timer_active: false,
        };

        Ok(rate_limiter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket_persistence() {
        let tb = TokenBucket::new(1000, 2000, 3000).unwrap();

        let restored_tb = TokenBucket::restore((), &tb.save()).unwrap();
        assert!(tb.partial_eq(&restored_tb));

        tb.reduce(100);
        let restored_tb = TokenBucket::restore((), &tb.save()).unwrap();
        assert!(tb.partial_eq(&restored_tb));

        tb.force_replenish(100);
        let restored_tb = TokenBucket::restore((), &tb.save()).unwrap();
        assert!(tb.partial_eq(&restored_tb));

        let tb_state = tb.save();
        let serialized_data = bitcode::serialize(&tb_state).unwrap();

        let restored_state = bitcode::deserialize(&serialized_data).unwrap();
        let restored_tb = TokenBucket::restore((), &restored_state).unwrap();
        assert!(tb.partial_eq(&restored_tb));
    }

    #[test]
    fn test_rate_limiter_persistence() {
        let refill_time = 100_000;
        let mut rate_limiter = RateLimiter::new(100, 0, refill_time, 10, 0, refill_time).unwrap();

        let restored_rate_limiter =
            RateLimiter::restore((), &rate_limiter.save()).expect("Unable to restore rate limiter");
        assert!(
            rate_limiter
                .ops()
                .unwrap()
                .partial_eq(restored_rate_limiter.ops().unwrap())
        );
        assert!(
            rate_limiter
                .bandwidth()
                .unwrap()
                .partial_eq(restored_rate_limiter.bandwidth().unwrap())
        );
        assert!(!restored_rate_limiter.timer_fd.is_armed());

        rate_limiter.consume(10, TokenType::Bytes);
        rate_limiter.consume(10, TokenType::Ops);
        let restored_rate_limiter =
            RateLimiter::restore((), &rate_limiter.save()).expect("Unable to restore rate limiter");
        assert!(
            rate_limiter
                .ops()
                .unwrap()
                .partial_eq(restored_rate_limiter.ops().unwrap())
        );
        assert!(
            rate_limiter
                .bandwidth()
                .unwrap()
                .partial_eq(restored_rate_limiter.bandwidth().unwrap())
        );
        assert!(!restored_rate_limiter.timer_fd.is_armed());

        rate_limiter.consume(1000, TokenType::Bytes);
        let restored_rate_limiter =
            RateLimiter::restore((), &rate_limiter.save()).expect("Unable to restore rate limiter");
        assert!(
            rate_limiter
                .ops()
                .unwrap()
                .partial_eq(restored_rate_limiter.ops().unwrap())
        );
        assert!(
            rate_limiter
                .bandwidth()
                .unwrap()
                .partial_eq(restored_rate_limiter.bandwidth().unwrap())
        );

        let rate_limiter_state = rate_limiter.save();
        let serialized_data = bitcode::serialize(&rate_limiter_state).unwrap();

        let restored_state = bitcode::deserialize(&serialized_data).unwrap();
        let restored_rate_limiter = RateLimiter::restore((), &restored_state).unwrap();

        assert!(
            rate_limiter
                .ops()
                .unwrap()
                .partial_eq(restored_rate_limiter.ops().unwrap())
        );
        assert!(
            rate_limiter
                .bandwidth()
                .unwrap()
                .partial_eq(restored_rate_limiter.bandwidth().unwrap())
        );
    }
}

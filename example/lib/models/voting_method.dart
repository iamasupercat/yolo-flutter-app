// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

/// Voting method for final inspection result
///
/// - [soft]: 평균 불량 확률이 0.5 이상이면 불량 (기본값)
/// - [hard]: 하나라도 불량이면 불량
enum VotingMethod { soft, hard }

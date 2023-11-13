// SPDX-License-Identifier: Apache-2.0
// This file is part of Frontier.
//
// Copyright (c) 2020-2022 Parity Technologies (UK) Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # EVM Pallet
//!
//! The EVM pallet allows unmodified EVM code to be executed in a Substrate-based blockchain.
//! - [`evm::Config`]
//!
//! ## EVM Engine
//!
//! The EVM pallet uses [`SputnikVM`](https://github.com/rust-blockchain/evm) as the underlying EVM engine.
//! The engine is overhauled so that it's [`modular`](https://github.com/corepaper/evm).
//!
//! ## Execution Lifecycle
//!
//! There are a separate set of accounts managed by the EVM pallet. Substrate based accounts can call the EVM Pallet
//! to deposit or withdraw balance from the Substrate base-currency into a different balance managed and used by
//! the EVM pallet. Once a user has populated their balance, they can create and call smart contracts using this pallet.
//!
//! There's one-to-one mapping from Substrate accounts and EVM external accounts that is defined by a conversion function.
//!
//! ## EVM Pallet vs Ethereum Network
//!
//! The EVM pallet should be able to produce nearly identical results compared to the Ethereum mainnet,
//! including gas cost and balance changes.
//!
//! Observable differences include:
//!
//! - The available length of block hashes may not be 256 depending on the configuration of the System pallet
//! in the Substrate runtime.
//! - Difficulty and coinbase, which do not make sense in this pallet and is currently hard coded to zero.
//!
//! We currently do not aim to make unobservable behaviors, such as state root, to be the same. We also don't aim to follow
//! the exact same transaction / receipt format. However, given one Ethereum transaction and one Substrate account's
//! private key, one should be able to convert any Ethereum transaction into a transaction compatible with this pallet.
//!
//! The gas configurations are configurable. Right now, a pre-defined London hard fork configuration option is provided.

// Ensure we're `no_std` when compiling for Wasm.
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(unused_crate_dependencies)]
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "runtime-benchmarks")]
pub mod benchmarking;

#[cfg(test)]
mod mock;
pub mod runner;
#[cfg(test)]
mod tests;
pub mod weights;

pub use evm::{
	Config as EvmConfig, Context, ExitError, ExitFatal, ExitReason, ExitRevert, ExitSucceed,
};
use hash_db::Hasher;
use impl_trait_for_tuples::impl_for_tuples;
use scale_codec::{Decode, Encode, MaxEncodedLen};
use scale_info::TypeInfo;
// Substrate
use frame_support::{
	dispatch::{DispatchResultWithPostInfo, Pays, PostDispatchInfo},
	storage::child::KillStorageResult,
	traits::{
		fungibles::Credit,
		tokens::{
			currency::Currency,
			fungible::Inspect,
			fungibles::{
				Balanced, Imbalance as FungiblesImbalance, Inspect as FungiblesInspect, Mutate,
			},
			Fortitude::Polite,
			Precision::Exact,
			Preservation::Protect,
		},
		FindAuthor, Get, Time,
	},
	weights::Weight,
};
use frame_system::RawOrigin;
use sp_core::{H160, H256, U256};
use sp_runtime::{
	traits::{BadOrigin, NumberFor, Saturating, UniqueSaturatedInto},
	AccountId32, DispatchErrorWithPostInfo,
};
use sp_std::{cmp::min, collections::btree_map::BTreeMap, vec::Vec};
// Frontier
use fp_account::AccountId20;
use fp_evm::GenesisAccount;
pub use fp_evm::{
	Account, CallInfo, CreateInfo, ExecutionInfoV2 as ExecutionInfo, FeeCalculator,
	IsPrecompileResult, LinearCostPrecompile, Log, Precompile, PrecompileFailure, PrecompileHandle,
	PrecompileOutput, PrecompileResult, PrecompileSet, TransactionValidationError, Vicinity,
};

pub use self::{
	pallet::*,
	runner::{Runner, RunnerError},
	weights::WeightInfo,
};

#[frame_support::pallet]
pub mod pallet {
	use super::*;
	use frame_support::pallet_prelude::*;
	use frame_system::pallet_prelude::*;

	#[pallet::pallet]
	#[pallet::without_storage_info]
	pub struct Pallet<T>(PhantomData<T>);

	#[pallet::config]
	pub trait Config: frame_system::Config {
		/// Calculator for current gas price.
		type FeeCalculator: FeeCalculator;

		/// Maps Ethereum gas to Substrate weight.
		type GasWeightMapping: GasWeightMapping;

		/// Weight corresponding to a gas unit.
		type WeightPerGas: Get<Weight>;

		/// Block number to block hash.
		type BlockHashMapping: BlockHashMapping;

		/// Allow the origin to call on behalf of given address.
		type CallOrigin: EnsureAddressOrigin<Self::RuntimeOrigin>;
		/// Allow the origin to withdraw on behalf of given address.
		type WithdrawOrigin: EnsureAddressOrigin<Self::RuntimeOrigin, Success = Self::AccountId>;

		/// Mapping from address to account id.
		type AddressMapping: AddressMapping<Self::AccountId>;
		/// Currency type for withdraw and balance storage.
		type Currency: Currency<Self::AccountId> + Inspect<Self::AccountId>;

		type Assets: Balanced<Self::AccountId>
			+ Mutate<Self::AccountId>
			+ FungiblesInspect<Self::AccountId>;

		/// The overarching event type.
		type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
		/// Precompiles associated with this EVM engine.
		type PrecompilesType: PrecompileSet;
		type PrecompilesValue: Get<Self::PrecompilesType>;
		/// Chain ID of EVM.
		type ChainId: Get<u64>;
		/// The block gas limit. Can be a simple constant, or an adjustment algorithm in another pallet.
		type BlockGasLimit: Get<U256>;
		/// EVM execution runner.
		type Runner: Runner<Self>;

		/// To handle fee deduction for EVM transactions. An example is this pallet being used by `pallet_ethereum`
		/// where the chain implementing `pallet_ethereum` should be able to configure what happens to the fees
		/// Similar to `OnChargeTransaction` of `pallet_transaction_payment`
		type OnChargeTransaction: OnChargeEVMTransaction<Self>;

		/// Called on create calls, used to record owner
		type OnCreate: OnCreate<Self>;

		/// Find author for the current block.
		type FindAuthor: FindAuthor<H160>;

		/// Gas limit Pov size ratio.
		type GasLimitPovSizeRatio: Get<u64>;

		/// Define the quick clear limit of storage clearing when a contract suicides. Set to 0 to disable it.
		type SuicideQuickClearLimit: Get<u32>;

		/// Get the timestamp for the current block.
		type Timestamp: Time;

		/// Weight information for extrinsics in this pallet.
		type WeightInfo: WeightInfo;

		/// EVM config used in the module.
		fn config() -> &'static EvmConfig {
			&SHANGHAI_CONFIG
		}
	}

	#[pallet::call]
	impl<T: Config> Pallet<T> {
		/// Withdraw balance from EVM into currency/balances pallet.
		#[pallet::call_index(0)]
		#[pallet::weight(<T as pallet::Config>::WeightInfo::withdraw())]
		pub fn withdraw(
			origin: OriginFor<T>,
			address: H160,
			value: AssetBalanceOf<T>,
		) -> DispatchResult {
			let destination = T::WithdrawOrigin::ensure_address_origin(&address, origin)?;
			let address_account_id = T::AddressMapping::into_account_id(address);
			let asset_id = NativeAssetId::<T>::get().unwrap();

			let _ =
				T::Assets::transfer(asset_id, &address_account_id, &destination, value, Protect);

			Ok(())
		}

		/// Issue an EVM call operation. This is similar to a message call transaction in Ethereum.
		#[pallet::call_index(1)]
		#[pallet::weight({
			let without_base_extrinsic_weight = true;
			T::GasWeightMapping::gas_to_weight(*gas_limit, without_base_extrinsic_weight)
		})]
		pub fn call(
			origin: OriginFor<T>,
			source: H160,
			target: H160,
			input: Vec<u8>,
			value: U256,
			gas_limit: u64,
			max_fee_per_gas: U256,
			max_priority_fee_per_gas: Option<U256>,
			nonce: Option<U256>,
			access_list: Vec<(H160, Vec<H256>)>,
		) -> DispatchResultWithPostInfo {
			T::CallOrigin::ensure_address_origin(&source, origin)?;

			let is_transactional = true;
			let validate = true;
			let info = match T::Runner::call(
				source,
				target,
				input,
				value,
				gas_limit,
				Some(max_fee_per_gas),
				max_priority_fee_per_gas,
				nonce,
				access_list,
				is_transactional,
				validate,
				None,
				None,
				T::config(),
			) {
				Ok(info) => info,
				Err(e) => {
					return Err(DispatchErrorWithPostInfo {
						post_info: PostDispatchInfo {
							actual_weight: Some(e.weight),
							pays_fee: Pays::Yes,
						},
						error: e.error.into(),
					})
				}
			};

			match info.exit_reason {
				ExitReason::Succeed(_) => {
					Pallet::<T>::deposit_event(Event::<T>::Executed { address: target });
				}
				_ => {
					Pallet::<T>::deposit_event(Event::<T>::ExecutedFailed { address: target });
				}
			};

			Ok(PostDispatchInfo {
				actual_weight: {
					let mut gas_to_weight = T::GasWeightMapping::gas_to_weight(
						info.used_gas.standard.unique_saturated_into(),
						true,
					);
					if let Some(weight_info) = info.weight_info {
						if let Some(proof_size_usage) = weight_info.proof_size_usage {
							*gas_to_weight.proof_size_mut() = proof_size_usage;
						}
					}
					Some(gas_to_weight)
				},
				pays_fee: Pays::No,
			})
		}

		/// Issue an EVM create operation. This is similar to a contract creation transaction in
		/// Ethereum.
		#[pallet::call_index(2)]
		#[pallet::weight({
			let without_base_extrinsic_weight = true;
			T::GasWeightMapping::gas_to_weight(*gas_limit, without_base_extrinsic_weight)
		})]
		pub fn create(
			origin: OriginFor<T>,
			source: H160,
			init: Vec<u8>,
			value: U256,
			gas_limit: u64,
			max_fee_per_gas: U256,
			max_priority_fee_per_gas: Option<U256>,
			nonce: Option<U256>,
			access_list: Vec<(H160, Vec<H256>)>,
		) -> DispatchResultWithPostInfo {
			T::CallOrigin::ensure_address_origin(&source, origin)?;

			let is_transactional = true;
			let validate = true;
			let info = match T::Runner::create(
				source,
				init,
				value,
				gas_limit,
				Some(max_fee_per_gas),
				max_priority_fee_per_gas,
				nonce,
				access_list,
				is_transactional,
				validate,
				None,
				None,
				T::config(),
			) {
				Ok(info) => info,
				Err(e) => {
					return Err(DispatchErrorWithPostInfo {
						post_info: PostDispatchInfo {
							actual_weight: Some(e.weight),
							pays_fee: Pays::Yes,
						},
						error: e.error.into(),
					})
				}
			};

			match info {
				CreateInfo {
					exit_reason: ExitReason::Succeed(_),
					value: create_address,
					..
				} => {
					Pallet::<T>::deposit_event(Event::<T>::Created {
						address: create_address,
					});
				}
				CreateInfo {
					exit_reason: _,
					value: create_address,
					..
				} => {
					Pallet::<T>::deposit_event(Event::<T>::CreatedFailed {
						address: create_address,
					});
				}
			}

			Ok(PostDispatchInfo {
				actual_weight: {
					let mut gas_to_weight = T::GasWeightMapping::gas_to_weight(
						info.used_gas.standard.unique_saturated_into(),
						true,
					);
					if let Some(weight_info) = info.weight_info {
						if let Some(proof_size_usage) = weight_info.proof_size_usage {
							*gas_to_weight.proof_size_mut() = proof_size_usage;
						}
					}
					Some(gas_to_weight)
				},
				pays_fee: Pays::No,
			})
		}

		/// Issue an EVM create2 operation.
		#[pallet::call_index(3)]
		#[pallet::weight({
			let without_base_extrinsic_weight = true;
			T::GasWeightMapping::gas_to_weight(*gas_limit, without_base_extrinsic_weight)
		})]
		pub fn create2(
			origin: OriginFor<T>,
			source: H160,
			init: Vec<u8>,
			salt: H256,
			value: U256,
			gas_limit: u64,
			max_fee_per_gas: U256,
			max_priority_fee_per_gas: Option<U256>,
			nonce: Option<U256>,
			access_list: Vec<(H160, Vec<H256>)>,
		) -> DispatchResultWithPostInfo {
			T::CallOrigin::ensure_address_origin(&source, origin)?;

			let is_transactional = true;
			let validate = true;
			let info = match T::Runner::create2(
				source,
				init,
				salt,
				value,
				gas_limit,
				Some(max_fee_per_gas),
				max_priority_fee_per_gas,
				nonce,
				access_list,
				is_transactional,
				validate,
				None,
				None,
				T::config(),
			) {
				Ok(info) => info,
				Err(e) => {
					return Err(DispatchErrorWithPostInfo {
						post_info: PostDispatchInfo {
							actual_weight: Some(e.weight),
							pays_fee: Pays::Yes,
						},
						error: e.error.into(),
					})
				}
			};

			match info {
				CreateInfo {
					exit_reason: ExitReason::Succeed(_),
					value: create_address,
					..
				} => {
					Pallet::<T>::deposit_event(Event::<T>::Created {
						address: create_address,
					});
				}
				CreateInfo {
					exit_reason: _,
					value: create_address,
					..
				} => {
					Pallet::<T>::deposit_event(Event::<T>::CreatedFailed {
						address: create_address,
					});
				}
			}

			Ok(PostDispatchInfo {
				actual_weight: {
					let mut gas_to_weight = T::GasWeightMapping::gas_to_weight(
						info.used_gas.standard.unique_saturated_into(),
						true,
					);
					if let Some(weight_info) = info.weight_info {
						if let Some(proof_size_usage) = weight_info.proof_size_usage {
							*gas_to_weight.proof_size_mut() = proof_size_usage;
						}
					}
					Some(gas_to_weight)
				},
				pays_fee: Pays::No,
			})
		}

		#[pallet::call_index(4)]
		#[pallet::weight(<T as pallet::Config>::WeightInfo::set_native_asset_id())]
		pub fn set_native_asset_id(origin: OriginFor<T>, asset_id: AssetIdOf<T>) -> DispatchResult {
			let _ = ensure_root(origin);

			NativeAssetId::<T>::put(asset_id.clone());

			Pallet::<T>::deposit_event(Event::<T>::NativeAssetIdChanged { asset_id });

			Ok(())
		}
	}

	#[pallet::event]
	#[pallet::generate_deposit(pub(super) fn deposit_event)]
	pub enum Event<T: Config> {
		/// Ethereum events from contracts.
		Log { log: Log },
		/// A contract has been created at given address.
		Created { address: H160 },
		/// A contract was attempted to be created, but the execution failed.
		CreatedFailed { address: H160 },
		/// A contract has been executed successfully with states applied.
		Executed { address: H160 },
		/// A contract has been executed with errors. States are reverted with only gas fees applied.
		ExecutedFailed { address: H160 },
		/// A contract has been executed with errors. States are reverted with only gas fees applied.
		NativeAssetIdChanged { asset_id: AssetIdOf<T> },
	}

	#[pallet::error]
	pub enum Error<T> {
		/// Not enough balance to perform action
		BalanceLow,
		/// Calculating total fee overflowed
		FeeOverflow,
		/// Calculating total payment overflowed
		PaymentOverflow,
		/// Withdraw fee failed
		WithdrawFailed,
		/// Gas price is too low.
		GasPriceTooLow,
		/// Nonce is invalid
		InvalidNonce,
		/// Gas limit is too low.
		GasLimitTooLow,
		/// Gas limit is too high.
		GasLimitTooHigh,
		/// The chain id is invalid.
		InvalidChainId,
		/// the signature is invalid.
		InvalidSignature,
		/// EVM reentrancy
		Reentrancy,
		/// EIP-3607,
		TransactionMustComeFromEOA,
		/// Undefined error.
		Undefined,
	}

	impl<T> From<TransactionValidationError> for Error<T> {
		fn from(validation_error: TransactionValidationError) -> Self {
			match validation_error {
				TransactionValidationError::GasLimitTooLow => Error::<T>::GasLimitTooLow,
				TransactionValidationError::GasLimitTooHigh => Error::<T>::GasLimitTooHigh,
				TransactionValidationError::BalanceTooLow => Error::<T>::BalanceLow,
				TransactionValidationError::TxNonceTooLow => Error::<T>::InvalidNonce,
				TransactionValidationError::TxNonceTooHigh => Error::<T>::InvalidNonce,
				TransactionValidationError::GasPriceTooLow => Error::<T>::GasPriceTooLow,
				TransactionValidationError::PriorityFeeTooHigh => Error::<T>::GasPriceTooLow,
				TransactionValidationError::InvalidFeeInput => Error::<T>::GasPriceTooLow,
				TransactionValidationError::InvalidChainId => Error::<T>::InvalidChainId,
				TransactionValidationError::InvalidSignature => Error::<T>::InvalidSignature,
				TransactionValidationError::UnknownError => Error::<T>::Undefined,
			}
		}
	}

	#[pallet::genesis_config]
	#[derive(frame_support::DefaultNoBound)]
	pub struct GenesisConfig<T> {
		pub accounts: BTreeMap<H160, GenesisAccount>,
		pub asset_id: u32,
		#[serde(skip)]
		pub _marker: PhantomData<T>,
	}

	#[pallet::genesis_build]
	impl<T: Config> BuildGenesisConfig for GenesisConfig<T>
	where
		U256: UniqueSaturatedInto<AssetBalanceOf<T>>,
		u32: scale_codec::EncodeLike<
			<<T as pallet::Config>::Assets as frame_support::traits::fungibles::Inspect<
				<T as frame_system::Config>::AccountId,
			>>::AssetId,
		>,
		<<T as pallet::Config>::Assets as frame_support::traits::fungibles::Inspect<
			<T as frame_system::Config>::AccountId,
		>>::AssetId: From<u32>,
	{
		fn build(&self) {
			const MAX_ACCOUNT_NONCE: usize = 100;
			NativeAssetId::<T>::put(self.asset_id);

			for (address, account) in &self.accounts {
				let account_id = T::AddressMapping::into_account_id(*address);

				// ASSUME: in one single EVM transaction, the nonce will not increase more than
				// `u128::max_value()`.
				for _ in 0..min(
					MAX_ACCOUNT_NONCE,
					UniqueSaturatedInto::<usize>::unique_saturated_into(account.nonce),
				) {
					frame_system::Pallet::<T>::inc_account_nonce(&account_id);
				}

				let _ = T::Assets::deposit(
					(self.asset_id).into(),
					&account_id,
					account.balance.unique_saturated_into(),
					Exact,
				);

				Pallet::<T>::create_account(*address, account.code.clone());

				for (index, value) in &account.storage {
					<AccountStorages<T>>::insert(address, index, value);
				}
			}
		}
	}

	#[pallet::storage]
	pub type NativeAssetId<T: Config> = StorageValue<_, AssetIdOf<T>, OptionQuery>;

	#[pallet::storage]
	pub type AccountCodes<T: Config> = StorageMap<_, Blake2_128Concat, H160, Vec<u8>, ValueQuery>;

	#[pallet::storage]
	pub type AccountCodesMetadata<T: Config> =
		StorageMap<_, Blake2_128Concat, H160, CodeMetadata, OptionQuery>;

	#[pallet::storage]
	pub type AccountStorages<T: Config> =
		StorageDoubleMap<_, Blake2_128Concat, H160, Blake2_128Concat, H256, H256, ValueQuery>;

	#[pallet::storage]
	pub type Suicided<T: Config> = StorageMap<_, Blake2_128Concat, H160, (), OptionQuery>;
}

/// Type alias for currency balance.
pub type BalanceOf<T> =
	<<T as Config>::Currency as Currency<<T as frame_system::Config>::AccountId>>::Balance;

// Type alias used for interaction with fungibles (assets).
// Balance type alias.
pub(crate) type AssetBalanceOf<T> =
	<<T as Config>::Assets as FungiblesInspect<<T as frame_system::Config>::AccountId>>::Balance;
/// Asset id type alias.
pub(crate) type AssetIdOf<T> =
	<<T as Config>::Assets as FungiblesInspect<<T as frame_system::Config>::AccountId>>::AssetId;

#[derive(
	Debug,
	Clone,
	Copy,
	Eq,
	PartialEq,
	Encode,
	Decode,
	TypeInfo,
	MaxEncodedLen
)]
pub struct CodeMetadata {
	pub size: u64,
	pub hash: H256,
}

impl CodeMetadata {
	fn from_code(code: &[u8]) -> Self {
		let size = code.len() as u64;
		let hash = H256::from(sp_io::hashing::keccak_256(code));

		Self { size, hash }
	}
}

pub trait EnsureAddressOrigin<OuterOrigin> {
	/// Success return type.
	type Success;

	/// Perform the origin check.
	fn ensure_address_origin(
		address: &H160,
		origin: OuterOrigin,
	) -> Result<Self::Success, BadOrigin> {
		Self::try_address_origin(address, origin).map_err(|_| BadOrigin)
	}

	/// Try with origin.
	fn try_address_origin(
		address: &H160,
		origin: OuterOrigin,
	) -> Result<Self::Success, OuterOrigin>;
}

/// Ensure that the EVM address is the same as the Substrate address. This only works if the account
/// ID is `H160`.
pub struct EnsureAddressSame;

impl<OuterOrigin> EnsureAddressOrigin<OuterOrigin> for EnsureAddressSame
where
	OuterOrigin: Into<Result<RawOrigin<H160>, OuterOrigin>> + From<RawOrigin<H160>>,
{
	type Success = H160;

	fn try_address_origin(address: &H160, origin: OuterOrigin) -> Result<H160, OuterOrigin> {
		origin.into().and_then(|o| match o {
			RawOrigin::Signed(who) if &who == address => Ok(who),
			r => Err(OuterOrigin::from(r)),
		})
	}
}

/// Ensure that the origin is root.
pub struct EnsureAddressRoot<AccountId>(sp_std::marker::PhantomData<AccountId>);

impl<OuterOrigin, AccountId> EnsureAddressOrigin<OuterOrigin> for EnsureAddressRoot<AccountId>
where
	OuterOrigin: Into<Result<RawOrigin<AccountId>, OuterOrigin>> + From<RawOrigin<AccountId>>,
{
	type Success = ();

	fn try_address_origin(_address: &H160, origin: OuterOrigin) -> Result<(), OuterOrigin> {
		origin.into().and_then(|o| match o {
			RawOrigin::Root => Ok(()),
			r => Err(OuterOrigin::from(r)),
		})
	}
}

/// Ensure that the origin never happens.
pub struct EnsureAddressNever<AccountId>(sp_std::marker::PhantomData<AccountId>);

impl<OuterOrigin, AccountId> EnsureAddressOrigin<OuterOrigin> for EnsureAddressNever<AccountId> {
	type Success = AccountId;

	fn try_address_origin(_address: &H160, origin: OuterOrigin) -> Result<AccountId, OuterOrigin> {
		Err(origin)
	}
}

/// Ensure that the address is truncated hash of the origin. Only works if the account id is
/// `AccountId32`.
pub struct EnsureAddressTruncated;

impl<OuterOrigin> EnsureAddressOrigin<OuterOrigin> for EnsureAddressTruncated
where
	OuterOrigin: Into<Result<RawOrigin<AccountId32>, OuterOrigin>> + From<RawOrigin<AccountId32>>,
{
	type Success = AccountId32;

	fn try_address_origin(address: &H160, origin: OuterOrigin) -> Result<AccountId32, OuterOrigin> {
		origin.into().and_then(|o| match o {
			RawOrigin::Signed(who) if AsRef::<[u8; 32]>::as_ref(&who)[0..20] == address[0..20] => {
				Ok(who)
			}
			r => Err(OuterOrigin::from(r)),
		})
	}
}

/// Ensure that the address is AccountId20.
pub struct EnsureAccountId20;

impl<OuterOrigin> EnsureAddressOrigin<OuterOrigin> for EnsureAccountId20
where
	OuterOrigin: Into<Result<RawOrigin<AccountId20>, OuterOrigin>> + From<RawOrigin<AccountId20>>,
{
	type Success = AccountId20;

	fn try_address_origin(address: &H160, origin: OuterOrigin) -> Result<AccountId20, OuterOrigin> {
		let acc: AccountId20 = AccountId20::from(*address);
		origin.into().and_then(|o| match o {
			RawOrigin::Signed(who) if who == acc => Ok(who),
			r => Err(OuterOrigin::from(r)),
		})
	}
}

/// Trait to be implemented for evm address mapping.
pub trait AddressMapping<A> {
	fn into_account_id(address: H160) -> A;
}

/// Identity address mapping.
pub struct IdentityAddressMapping;

impl<T: From<H160>> AddressMapping<T> for IdentityAddressMapping {
	fn into_account_id(address: H160) -> T {
		address.into()
	}
}

/// Hashed address mapping.
pub struct HashedAddressMapping<H>(sp_std::marker::PhantomData<H>);

impl<H: Hasher<Out = H256>> AddressMapping<AccountId32> for HashedAddressMapping<H> {
	fn into_account_id(address: H160) -> AccountId32 {
		let mut data = [0u8; 24];
		data[0..4].copy_from_slice(b"evm:");
		data[4..24].copy_from_slice(&address[..]);
		let hash = H::hash(&data);

		AccountId32::from(Into::<[u8; 32]>::into(hash))
	}
}

/// A trait for getting a block hash by number.
pub trait BlockHashMapping {
	fn block_hash(number: u32) -> H256;
}

/// Returns the Substrate block hash by number.
pub struct SubstrateBlockHashMapping<T>(sp_std::marker::PhantomData<T>);
impl<T: Config> BlockHashMapping for SubstrateBlockHashMapping<T> {
	fn block_hash(number: u32) -> H256 {
		let number = <NumberFor<T::Block>>::from(number);
		H256::from_slice(frame_system::Pallet::<T>::block_hash(number).as_ref())
	}
}

/// A mapping function that converts Ethereum gas to Substrate weight
pub trait GasWeightMapping {
	fn gas_to_weight(gas: u64, without_base_weight: bool) -> Weight;
	fn weight_to_gas(weight: Weight) -> u64;
}

pub struct FixedGasWeightMapping<T>(sp_std::marker::PhantomData<T>);
impl<T: Config> GasWeightMapping for FixedGasWeightMapping<T> {
	fn gas_to_weight(gas: u64, without_base_weight: bool) -> Weight {
		let mut weight = T::WeightPerGas::get().saturating_mul(gas);
		if without_base_weight {
			weight = weight.saturating_sub(
				T::BlockWeights::get()
					.get(frame_support::dispatch::DispatchClass::Normal)
					.base_extrinsic,
			);
		}
		// Apply a gas to proof size ratio based on BlockGasLimit
		let ratio = T::GasLimitPovSizeRatio::get();
		if ratio > 0 {
			let proof_size = gas.saturating_div(ratio);
			*weight.proof_size_mut() = proof_size;
		}

		weight
	}
	fn weight_to_gas(weight: Weight) -> u64 {
		weight.div(T::WeightPerGas::get().ref_time()).ref_time()
	}
}

static SHANGHAI_CONFIG: EvmConfig = EvmConfig::shanghai();

impl<T: Config> Pallet<T> {
	/// Check whether an account is empty.
	pub fn is_account_empty(address: &H160) -> bool {
		let (account, _) = Self::account_basic(address);
		let code_len = <AccountCodes<T>>::decode_len(address).unwrap_or(0);

		account.nonce == U256::zero() && account.balance == U256::zero() && code_len == 0
	}

	/// Remove an account if its empty.
	pub fn remove_account_if_empty(address: &H160) {
		if Self::is_account_empty(address) {
			Self::remove_account(address);
		}
	}

	/// Remove an account.
	pub fn remove_account(address: &H160) {
		if <AccountCodes<T>>::contains_key(address) {
			// Remember to call `dec_sufficients` when clearing Suicided.
			<Suicided<T>>::insert(address, ());

			// In theory, we can always have pre-EIP161 contracts, so we
			// make sure the account nonce is at least one.
			let account_id = T::AddressMapping::into_account_id(*address);
			frame_system::Pallet::<T>::inc_account_nonce(&account_id);
		}

		<AccountCodes<T>>::remove(address);
		<AccountCodesMetadata<T>>::remove(address);

		if T::SuicideQuickClearLimit::get() > 0 {
			#[allow(deprecated)]
			let res = <AccountStorages<T>>::remove_prefix(address, Some(T::SuicideQuickClearLimit::get()));

			match res {
				KillStorageResult::AllRemoved(_) => {
					<Suicided<T>>::remove(address);

					let account_id = T::AddressMapping::into_account_id(*address);
					let _ = frame_system::Pallet::<T>::dec_sufficients(&account_id);
				}
				KillStorageResult::SomeRemaining(_) => (),
			}
		}
	}

	/// Create an account.
	pub fn create_account(address: H160, code: Vec<u8>) {
		if <Suicided<T>>::contains_key(address) {
			// This branch should never trigger, because when Suicided
			// contains an address, then its nonce will be at least one,
			// which causes CreateCollision error in EVM, but we add it
			// here for safeguard.
			return;
		}

		if code.is_empty() {
			return;
		}

		if !<AccountCodes<T>>::contains_key(address) {
			let account_id = T::AddressMapping::into_account_id(address);
			let _ = frame_system::Pallet::<T>::inc_sufficients(&account_id);
		}

		// Update metadata.
		let meta = CodeMetadata::from_code(&code);
		<AccountCodesMetadata<T>>::insert(address, meta);

		<AccountCodes<T>>::insert(address, code);
	}

	/// Get the account metadata (hash and size) from storage if it exists,
	/// or compute it from code and store it if it doesn't exist.
	pub fn account_code_metadata(address: H160) -> CodeMetadata {
		if let Some(meta) = <AccountCodesMetadata<T>>::get(address) {
			return meta;
		}

		let code = <AccountCodes<T>>::get(address);

		// If code is empty we return precomputed hash for empty code.
		// We don't store it as this address could get code deployed in the future.
		if code.is_empty() {
			const EMPTY_CODE_HASH: [u8; 32] = hex_literal::hex!(
				"c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
			);
			return CodeMetadata {
				size: 0,
				hash: EMPTY_CODE_HASH.into(),
			};
		}

		let meta = CodeMetadata::from_code(&code);

		<AccountCodesMetadata<T>>::insert(address, meta);
		meta
	}

	/// Get the account basic in EVM format.
	pub fn account_basic(address: &H160) -> (Account, frame_support::weights::Weight) {
		let account_id = T::AddressMapping::into_account_id(*address);
		let asset_id = NativeAssetId::<T>::get().unwrap();

		let nonce = frame_system::Pallet::<T>::account_nonce(&account_id);
		// keepalive `true` takes into account ExistentialDeposit as part of what's considered liquid balance.
		let balance = T::Assets::reducible_balance(asset_id, &account_id, Protect, Polite);

		(
			Account {
				nonce: U256::from(UniqueSaturatedInto::<u128>::unique_saturated_into(nonce)),
				balance: U256::from(UniqueSaturatedInto::<u128>::unique_saturated_into(balance)),
			},
			T::DbWeight::get().reads(2),
		)
	}

	/// Get the author using the FindAuthor trait.
	pub fn find_author() -> H160 {
		let digest = <frame_system::Pallet<T>>::digest();
		let pre_runtime_digests = digest.logs.iter().filter_map(|d| d.as_pre_runtime());

		T::FindAuthor::find_author(pre_runtime_digests).unwrap_or_default()
	}
}

/// Handle withdrawing, refunding and depositing of transaction fees.
/// Similar to `OnChargeTransaction` of `pallet_transaction_payment`
pub trait OnChargeEVMTransaction<T: Config> {
	type LiquidityInfo: Default;

	/// Before the transaction is executed the payment of the transaction fees
	/// need to be secured.
	fn withdraw_fee(who: &H160, fee: U256) -> Result<Self::LiquidityInfo, Error<T>>;

	/// After the transaction was executed the actual fee can be calculated.
	/// This function should refund any overpaid fees and optionally deposit
	/// the corrected amount, and handles the base fee rationing using the provided
	/// `OnUnbalanced` implementation.
	/// Returns the `NegativeImbalance` - if any - produced by the priority fee.
	fn correct_and_deposit_fee(
		who: &H160,
		corrected_fee: U256,
		base_fee: U256,
		already_withdrawn: Self::LiquidityInfo,
	) -> Self::LiquidityInfo;

	/// Introduced in EIP1559 to handle the priority tip.
	fn pay_priority_fee(tip: Self::LiquidityInfo);
}

/// Implements the transaction payment for a pallet implementing the `Currency`
/// trait (eg. the pallet_balances) using an unbalance handler (implementing
/// `OnUnbalanced`).
/// Similar to `CurrencyAdapter` of `pallet_transaction_payment`
pub struct EVMAssetsAdapter<T: Config>(sp_std::marker::PhantomData<T>);

impl<T> OnChargeEVMTransaction<T> for EVMAssetsAdapter<T>
where
	T: Config,
	U256: UniqueSaturatedInto<AssetBalanceOf<T>>,
{
	// Kept type as Option to satisfy bound of Default
	type LiquidityInfo = Option<Credit<T::AccountId, T::Assets>>;

	fn withdraw_fee(who: &H160, fee: U256) -> Result<Self::LiquidityInfo, Error<T>> {
		if fee.is_zero() {
			return Ok(None);
		}
		let asset_id = NativeAssetId::<T>::get().unwrap();
		let account_id = T::AddressMapping::into_account_id(*who);
		let imbalance = T::Assets::withdraw(
			asset_id,
			&account_id,
			fee.unique_saturated_into(),
			Exact,
			Protect,
			Polite,
		)
		.map_err(|_| Error::<T>::BalanceLow)?;
		Ok(Some(imbalance))
	}

	fn correct_and_deposit_fee(
		who: &H160,
		corrected_fee: U256,
		_base_fee: U256,
		already_withdrawn: Self::LiquidityInfo,
	) -> Self::LiquidityInfo {
		if let Some(paid) = already_withdrawn {
			let account_id = T::AddressMapping::into_account_id(*who);
			let asset_id = NativeAssetId::<T>::get().unwrap();
			let zero = FungiblesImbalance::zero(asset_id.clone());

			// Calculate how much refund we should return
			let refund_amount = paid
				.peek()
				.saturating_sub(corrected_fee.unique_saturated_into());
			// refund to the account that paid the fees. If this fails, the
			// account might have dropped below the existential balance. In
			// that case we don't refund anything.
			let _refund_imbalance = T::Assets::deposit(asset_id, &account_id, refund_amount, Exact)
				.unwrap_or_else(|_| zero);

			return None;
		}
		None
	}

	fn pay_priority_fee(tip: Self::LiquidityInfo) {
		// Default Ethereum behaviour: issue the tip to the block author.
		if let Some(tip) = tip {
			let account_id = T::AddressMapping::into_account_id(<Pallet<T>>::find_author());
			let asset_id = NativeAssetId::<T>::get().unwrap();
			let _ = T::Assets::deposit(asset_id, &account_id, tip.peek(), Exact);
		}
	}
}

/// Implementation for () does not specify what to do with imbalance
impl<T> OnChargeEVMTransaction<T> for ()
where
	T: Config,
	U256: UniqueSaturatedInto<AssetBalanceOf<T>>,
{
	// Kept type as Option to satisfy bound of Default
	type LiquidityInfo = Option<Credit<T::AccountId, T::Assets>>;

	fn withdraw_fee(who: &H160, fee: U256) -> Result<Self::LiquidityInfo, Error<T>> {
		EVMAssetsAdapter::<T>::withdraw_fee(who, fee)
	}

	fn correct_and_deposit_fee(
		who: &H160,
		corrected_fee: U256,
		base_fee: U256,
		already_withdrawn: Self::LiquidityInfo,
	) -> Self::LiquidityInfo {
		EVMAssetsAdapter::<T>::correct_and_deposit_fee(
			who,
			corrected_fee,
			base_fee,
			already_withdrawn,
		)
	}

	fn pay_priority_fee(tip: Self::LiquidityInfo) {
		EVMAssetsAdapter::<T>::pay_priority_fee(tip)
	}
}

pub trait OnCreate<T> {
	fn on_create(owner: H160, contract: H160);
}

impl<T> OnCreate<T> for () {
	fn on_create(_owner: H160, _contract: H160) {}
}

#[impl_for_tuples(1, 12)]
impl<T> OnCreate<T> for Tuple {
	fn on_create(owner: H160, contract: H160) {
		for_tuples!(#(
			Tuple::on_create(owner, contract);
		)*)
	}
}

/**
 * @file   tm.cpp
 * @author Božo Đerek <bozo.derek@epfl.ch>
**/

// External headers
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

// Internal headers
#include <tm.hpp>

#include "macros.hpp"

constexpr size_t ENTRIES = 1 << 20;
constexpr uintptr_t MASK = 0x3FFFFC;

/**
 * @brief Simple Shared Memory Region (a.k.a Transactional Memory).
 */
struct region {
    std::atomic<uint32_t> global_version_clock;
    std::mutex allocs_mtx;
    std::atomic<uint32_t> vwls[ENTRIES];

    void* start;               // Start of the shared memory region (i.e., of the non-deallocable memory segment)
    std::vector<void*> allocs; // Shared memory segments dynamically allocated via tm_alloc within transactions
    size_t size;               // Size of the non-deallocable memory segment (in bytes)
    size_t align;              // Size of a word in the shared memory region (in bytes)
};

thread_local struct transaction {
    bool is_ro;
    uint32_t rv;
    uint32_t wv;
    std::vector<void*> read_set;
    std::unordered_map<void*, void*> write_set;
} txn;

inline std::atomic<uint32_t>& vwl_get(region* rgn, const void* addr) {
    return rgn->vwls[(uintptr_t) addr & MASK];
}

inline bool vwl_lock(std::atomic<uint32_t>& vwl) {
    auto lck = vwl.load();

    return !(lck & 0b1) && vwl.compare_exchange_strong(lck, lck | 0b1);
}

inline void vwl_unlock(std::atomic<uint32_t>& vwl) {
    vwl.fetch_and(~((uint32_t) 1));
}

inline void vwl_set_version_and_unlock(std::atomic<uint32_t>& vwl) {
    vwl.store(txn.wv << 1);
}

inline std::pair<uint32_t, bool> vwl_read(std::atomic<uint32_t>& vwl) {
    auto lck = vwl.load();

    return {lck >> 1, lck & 0b1};
}

inline bool vwl_check(std::atomic<uint32_t>& vwl) {
    auto [version, lck_bit] = vwl_read(vwl);

    return !lck_bit && version <= txn.rv;
}

inline bool validate_read_set(region* rgn) {
    return std::all_of(txn.read_set.cbegin(), txn.read_set.cend(), [&](const auto& addr) {
        return vwl_check(vwl_get(rgn, addr));
    });
}

inline bool lock_write_set(region* rgn) {
    std::vector<void*> locked_addrs;
    for (const auto& [addr, _] : txn.write_set) {
        if (!vwl_lock(vwl_get(rgn, addr))) {
            for (const auto& locked_addr : locked_addrs) {
                vwl_unlock(vwl_get(rgn, locked_addr));
            }
            return false;
        }
        locked_addrs.push_back(addr);
    }

    return true;
}

inline void unlock_write_set(region* rgn) {
    for (const auto& [addr, _] : txn.write_set) {
        vwl_unlock(vwl_get(rgn, addr));
    }
}

inline void free_write_set(void) {
    for (const auto& [_, val] : txn.write_set) {
        std::free(val);
    }
}

inline void commit_and_release_locks(region* rgn) {
    for (const auto& [addr, val] : txn.write_set) {
        std::memcpy(addr, val, rgn->align);
        vwl_set_version_and_unlock(vwl_get(rgn, addr));
    }
}

inline bool commit(region* rgn) {
    if (!lock_write_set(rgn)) {
        return false;
    }

    // increment global version-clock
    txn.wv = 1 + rgn->global_version_clock.fetch_add(1);
    if (txn.rv + 1 != txn.wv && !validate_read_set(rgn)) {
        unlock_write_set(rgn);
        return false;
    }

    commit_and_release_locks(rgn);
    return true;
}


/** Create (i.e. allocate + init) a new shared memory region, with one first non-free-able allocated segment of the requested size and alignment.
 * @param size  Size of the first shared segment of memory to allocate (in bytes), must be a positive multiple of the alignment
 * @param align Alignment (in bytes, must be a power of 2) that the shared memory region must support
 * @return Opaque shared memory region handle, 'invalid_shared' on failure
**/
shared_t tm_create(size_t size, size_t align) noexcept {
    auto rgn = new (std::nothrow) region{};
    if (unlikely(!rgn)) {
        return invalid_shared;
    }

    rgn->start = std::aligned_alloc(align, size);
    if (unlikely(!rgn->start)) {
        delete rgn;
        return invalid_shared;
    }
    std::memset(rgn->start, 0, size);

    rgn->global_version_clock.store(0);
    rgn->size = size;
    rgn->align = align;

    return (shared_t) rgn;
}

/** Destroy (i.e. clean-up + free) a given shared memory region.
 * @param shared Shared memory region to destroy, with no running transaction
**/
void tm_destroy(shared_t shared) noexcept {
    auto rgn = (region*) shared;

    std::free(rgn->start);

    for (const auto& alloc : rgn->allocs) {
        std::free(alloc);
    }
    
    delete rgn;
}

/** [thread-safe] Return the start address of the first allocated segment in the shared memory region.
 * @param shared Shared memory region to query
 * @return Start address of the first allocated segment
**/
void* tm_start(shared_t shared) noexcept {
    return ((region*) shared)->start;
}

/** [thread-safe] Return the size (in bytes) of the first allocated segment of the shared memory region.
 * @param shared Shared memory region to query
 * @return First allocated segment size
**/
size_t tm_size(shared_t shared) noexcept {
    return ((region*) shared)->size;
}

/** [thread-safe] Return the alignment (in bytes) of the memory accesses on the given shared memory region.
 * @param shared Shared memory region to query
 * @return Alignment used globally
**/
size_t tm_align(shared_t shared) noexcept {
    return ((region*) shared)->align;
}

/** [thread-safe] Begin a new transaction on the given shared memory region.
 * @param shared Shared memory region to start a transaction on
 * @param is_ro  Whether the transaction is read-only
 * @return Opaque transaction ID, 'invalid_tx' on failure
**/
tx_t tm_begin(shared_t shared, bool is_ro) noexcept {
    txn.is_ro = is_ro;
    // sample global version-clock
    txn.rv = ((region*) shared)->global_version_clock.load();
    txn.wv = 0;
    txn.read_set.clear();
    txn.write_set.clear();
    
    return (tx_t) &txn;
}

/** [thread-safe] End the given transaction.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to end
 * @return Whether the whole transaction committed
**/
bool tm_end(shared_t shared, tx_t tx [[maybe_unused]]) noexcept {
   if (txn.is_ro) {
        return true;
    }

    auto committed = commit((region*) shared);
    free_write_set();
    return committed;
}

/** [thread-safe] Read operation in the given transaction, source in the shared region and target in a private region.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param source Source start address (in the shared region)
 * @param size   Length to copy (in bytes), must be a positive multiple of the alignment
 * @param target Target start address (in a private region)
 * @return Whether the whole transaction can continue
**/
bool tm_read(shared_t shared, tx_t tx [[maybe_unused]], void const* source, size_t size, void* target) noexcept {
    auto rgn = (region*) shared;
    auto align = rgn->align;

    if (txn.is_ro) {
        for (size_t i = 0; i < size; i += align) {
            void* src_addr = (char*) source + i;
            void* trgt_addr = (char*) target + i;
            
            auto& vwl = vwl_get(rgn, src_addr);
            
            auto [version, lck_bit] = vwl_read(vwl);
            if (lck_bit || version > txn.rv) {
                return false;
            }

            std::memcpy(trgt_addr, src_addr, align);

            auto [new_version, new_lck_bit] = vwl_read(vwl);
            if (new_lck_bit || version != new_version) {
                return false;
            }
        }
    } else {
        for (size_t i = 0; i < size; i += align) {
            void* src_addr = (char*) source + i;
            void* trgt_addr = (char*) target + i;

            auto it = txn.write_set.find(src_addr);
            if (it != txn.write_set.end()) {
                std::memcpy(trgt_addr, it->second, align);
                continue;
            }

            auto& vwl = vwl_get(rgn, src_addr);
            
            auto [version, lck_bit] = vwl_read(vwl);
            if (lck_bit || version > txn.rv) {
                free_write_set();
                return false;
            }

            std::memcpy(trgt_addr, src_addr, align);

            auto [new_version, new_lck_bit] = vwl_read(vwl);
            if (new_lck_bit || version != new_version) {
                free_write_set();
                return false;
            }

            txn.read_set.push_back(src_addr);
        }
    }

    return true;
}

/** [thread-safe] Write operation in the given transaction, source in a private region and target in the shared region.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param source Source start address (in a private region)
 * @param size   Length to copy (in bytes), must be a positive multiple of the alignment
 * @param target Target start address (in the shared region)
 * @return Whether the whole transaction can continue
**/
bool tm_write(shared_t shared, tx_t tx [[maybe_unused]], void const* source, size_t size, void* target) noexcept {
    auto align = ((region*) shared)->align;

    for (size_t i = 0; i < size; i += align) {
        void* src_addr = (char*) source + i;
        void* trgt_addr = (char*) target + i;

        if (txn.write_set.find(trgt_addr) == txn.write_set.end()) {
            txn.write_set[trgt_addr] = std::malloc(align);
            if (unlikely(!txn.write_set[trgt_addr])) {
                free_write_set();
                return false;
            }
        }
        std::memcpy(txn.write_set[trgt_addr], src_addr, align);
    }

    return true;
}

/** [thread-safe] Memory allocation in the given transaction.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param size   Allocation requested size (in bytes), must be a positive multiple of the alignment
 * @param target Pointer in private memory receiving the address of the first byte of the newly allocated, aligned segment
 * @return Whether the whole transaction can continue (success/nomem), or not (abort_alloc)
**/
Alloc tm_alloc(shared_t shared, tx_t tx [[maybe_unused]], size_t size, void** target) noexcept {
    auto rgn = (region*) shared;

    auto sgmnt = std::aligned_alloc(std::max(rgn->align, sizeof(void*)), size);
    if (unlikely(!sgmnt)) {
        return Alloc::nomem;
    }

    {
        std::lock_guard<std::mutex> lock(rgn->allocs_mtx);
        rgn->allocs.push_back(sgmnt);
    }

    std::memset(sgmnt, 0, size);
    *target = sgmnt;

    return Alloc::success;
}

/** [thread-safe] Memory freeing in the given transaction.
 * @param shared Shared memory region associated with the transaction
 * @param tx     Transaction to use
 * @param target Address of the first byte of the previously allocated segment to deallocate
 * @return Whether the whole transaction can continue
**/
bool tm_free(shared_t shared [[maybe_unused]], tx_t tx [[maybe_unused]], void* target [[maybe_unused]]) noexcept {
    return true;
}

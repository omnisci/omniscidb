/*
 * Copyright 2020 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#if !TBB_PREVIEW_ISOLATED_TASK_GROUP
#error Must define TBB_PREVIEW_ISOLATED_TASK_GROUP
#endif
#include "tbb/concurrent_vector.h"
#include "tbb/task_arena.h"
#include "tbb/task_group.h"

#include <future>
#include <iostream>
#include <type_traits>

#include "Utils/Threading.h"

namespace threadpool {

template <typename T>
class FuturesThreadPoolBase {
 public:
  template <class Function, class... Args>
  void spawn(Function&& f, Args&&... args) {
    threads_.push_back(
        utils::async(std::forward<Function>(f), std::forward<Args>(args)...));
  }

 protected:
  std::vector<utils::future<T>> threads_;
};

template <typename T, typename ENABLE = void>
class FuturesThreadPool : public FuturesThreadPoolBase<T> {
 public:
  FuturesThreadPool() {}

  void join() {
    for (auto& child : this->threads_) {
      child.wait();
    }
  }
};

template <typename T>
class FuturesThreadPool<T, std::enable_if_t<std::is_object<T>::value>>
    : public FuturesThreadPoolBase<T> {
 public:
  FuturesThreadPool() {}

  auto join() {
    std::vector<T> results;
    results.reserve(this->threads_.size());
    for (auto& child : this->threads_) {
      child.wait();
    }
    for (auto& child : this->threads_) {
      results.push_back(child.get());
    }
    return results;
  }
};

class TbbThreadPoolBase {
 protected:
  tbb::isolated_task_group tasks_;
};

template <typename T, typename ENABLE = void>
class TbbThreadPool : public TbbThreadPoolBase {
 public:
  TbbThreadPool() {}

  template <class Function, class... Args>
  void spawn(Function&& f, Args&&... args) {
    utils::g_tbb_arena.execute([&, this] { tasks_.run([f, args...] { f(args...); }); });
  }
  //TODO: append?

  void join() {
    utils::g_tbb_arena.execute([&, this] { tasks_.wait(); });
  }
};

template <typename T>
class TbbThreadPool<T, std::enable_if_t<std::is_object<T>::value>>
    : public TbbThreadPoolBase {
 public:
  TbbThreadPool() {}

  template <class Function, class... Args>
  void spawn(Function&& f, Args&&... args) {
    const size_t result_idx = results_.size();
    results_.emplace_back(
        T{});  // TODO: there is a race between accessing the vector and resizing
    utils::g_tbb_arena.execute([&, this] {
      tasks_.run([this, result_idx, f, args...] { results_[result_idx] = f(args...); });
    });
  }

  auto join() {
    utils::g_tbb_arena.execute([&, this] { tasks_.wait(); });
    return results_;
  }

 private:
  std::vector<T> results_;
};

#ifdef ENABLE_TBB
template <typename T>
using ThreadPool = TbbThreadPool<T>;
#else
template <typename T>
using ThreadPool = FuturesThreadPool<T>;
#endif
};  // namespace threadpool

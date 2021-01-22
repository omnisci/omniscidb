/*
 * Copyright 2019 OmniSci, Inc.
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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <type_traits>
#include <vector>

#include "Utils/Threading.h"

namespace ThreadController_NS {

template <typename FutureReturnType>
struct FutureGetter {
  using FutureGetterFunction = void (*)(FutureReturnType);
  FutureGetterFunction get;
};

template <>
struct FutureGetter<void> {};

template <typename FutureReturnType = void>
class SimpleThreadController {
 public:
  SimpleThreadController() = delete;
  template <bool future_return_void = std::is_void<FutureReturnType>::value>
  SimpleThreadController(const int max_threads, std::enable_if_t<future_return_void>* = 0)
      : max_threads_(max_threads) {}
  template <bool future_return_void = std::is_void<FutureReturnType>::value>
  SimpleThreadController(const int max_threads,
                         const FutureGetter<FutureReturnType> future_getter,
                         std::enable_if_t<!future_return_void>* = 0)
      : max_threads_(max_threads), future_getter_(future_getter) {}
  virtual ~SimpleThreadController() {}
  virtual int getThreadCount() const { return threads_.size(); }
  virtual int getRunningThreadCount() const { return threads_.size(); }
  virtual void checkThreadsStatus() {
    while (getRunningThreadCount() >= max_threads_) {
      std::this_thread::yield();
      threads_.erase(std::remove_if(threads_.begin(),
                                    threads_.end(),
                                    [this](auto& th) {
                                      using namespace std::chrono_literals;
                                      if (th.wait_for(0ns) == std::future_status::ready) {
                                        this->get_future(th);
                                        return true;
                                      } else {
                                        return false;
                                      }
                                    }),
                     threads_.end());
    }
  }
  template <typename FuncType, typename... Args>
  void startThread(FuncType&& func, Args&&... args) {
    threads_.emplace_back(std::async(std::launch::async, func, args...));
  }
  virtual void finish() {
    for (auto& t : threads_) {
      get_future(t);
    }
    threads_.clear();
  }

 protected:
  template <bool future_return_void = std::is_void<FutureReturnType>::value>
  void get_future(std::future<FutureReturnType>& future,
                  std::enable_if_t<future_return_void>* = 0) {
    future.get();
  }
  template <bool future_return_void = std::is_void<FutureReturnType>::value>
  void get_future(std::future<FutureReturnType>& future,
                  std::enable_if_t<!future_return_void>* = 0) {
    future_getter_.get(future.get());
  }

 private:
  const int max_threads_;
  const FutureGetter<FutureReturnType> future_getter_{};
  std::vector<std::future<FutureReturnType>> threads_;
};

template <typename FutureReturnType = void>
class SimpleRunningThreadController : public SimpleThreadController<FutureReturnType> {
 public:
  SimpleRunningThreadController() = delete;
  template <bool future_return_void = std::is_void<FutureReturnType>::value>
  SimpleRunningThreadController(const int max_threads,
                                std::enable_if_t<future_return_void>* = 0)
      : SimpleThreadController<FutureReturnType>(max_threads), n_running_threads_(0) {}
  template <bool future_return_void = std::is_void<FutureReturnType>::value>
  SimpleRunningThreadController(const int max_threads,
                                const FutureGetter<FutureReturnType> future_getter,
                                std::enable_if_t<!future_return_void>* = 0)
      : SimpleThreadController<FutureReturnType>(max_threads, future_getter)
      , n_running_threads_(0) {}
  ~SimpleRunningThreadController() override {}
  int notify_thread_is_completed() { return --n_running_threads_; }
  int getRunningThreadCount() const override { return n_running_threads_; }
  void checkThreadsStatus() override {
    SimpleThreadController<FutureReturnType>::checkThreadsStatus();
  }
  template <typename FuncType, typename... Args>
  int startThread(FuncType&& func, Args&&... args) {
    SimpleThreadController<FutureReturnType>::startThread(func, args...);
    return ++n_running_threads_;
  }

 private:
  // SimpleRunningThreadController consumers must EXPLICITLY update number
  // of running threads using notify_thread_is_completed member function
  std::atomic<int> n_running_threads_;
};

}  // namespace ThreadController_NS

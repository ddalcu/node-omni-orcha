#pragma once
#include <napi.h>
#include <functional>
#include <string>

/**
 * Generic AsyncWorker that runs a function on a background thread
 * and resolves a Promise with the result on the main thread.
 */
template <typename TResult>
class PromiseWorker : public Napi::AsyncWorker {
public:
  using WorkFn = std::function<TResult()>;
  using ResolveFn = std::function<Napi::Value(Napi::Env, const TResult&)>;

  PromiseWorker(
    Napi::Env env,
    WorkFn work,
    ResolveFn resolve
  ) : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      work_(std::move(work)),
      resolve_(std::move(resolve)) {}

  Napi::Promise Promise() { return deferred_.Promise(); }

protected:
  void Execute() override {
    try {
      result_ = work_();
    } catch (const std::exception& e) {
      SetError(e.what());
    }
  }

  void OnOK() override {
    auto env = Env();
    Napi::HandleScope scope(env);
    try {
      deferred_.Resolve(resolve_(env, result_));
    } catch (const Napi::Error& e) {
      deferred_.Reject(e.Value());
    }
  }

  void OnError(const Napi::Error& e) override {
    deferred_.Reject(e.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
  WorkFn work_;
  ResolveFn resolve_;
  TResult result_;
};

/**
 * Simplified async worker for void operations (e.g., model loading).
 */
class VoidPromiseWorker : public Napi::AsyncWorker {
public:
  using WorkFn = std::function<void()>;

  VoidPromiseWorker(Napi::Env env, WorkFn work)
    : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      work_(std::move(work)) {}

  Napi::Promise Promise() { return deferred_.Promise(); }

protected:
  void Execute() override {
    try {
      work_();
    } catch (const std::exception& e) {
      SetError(e.what());
    }
  }

  void OnOK() override {
    deferred_.Resolve(Env().Undefined());
  }

  void OnError(const Napi::Error& e) override {
    deferred_.Reject(e.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
  WorkFn work_;
};

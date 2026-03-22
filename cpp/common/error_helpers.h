#pragma once
#include <napi.h>
#include <string>

inline Napi::Error createError(Napi::Env env, const std::string& message) {
  return Napi::Error::New(env, message);
}

inline void throwIfNull(Napi::Env env, const void* ptr, const std::string& message) {
  if (!ptr) {
    throw createError(env, message);
  }
}

inline std::string getStringArg(const Napi::CallbackInfo& info, size_t index, const std::string& name) {
  if (index >= info.Length() || !info[index].IsString()) {
    throw Napi::TypeError::New(info.Env(), name + " must be a string");
  }
  return info[index].As<Napi::String>().Utf8Value();
}

inline int32_t getInt32Option(Napi::Object obj, const std::string& key, int32_t defaultVal) {
  if (obj.Has(key) && obj.Get(key).IsNumber()) {
    return obj.Get(key).As<Napi::Number>().Int32Value();
  }
  return defaultVal;
}

inline double getDoubleOption(Napi::Object obj, const std::string& key, double defaultVal) {
  if (obj.Has(key) && obj.Get(key).IsNumber()) {
    return obj.Get(key).As<Napi::Number>().DoubleValue();
  }
  return defaultVal;
}

inline bool getBoolOption(Napi::Object obj, const std::string& key, bool defaultVal) {
  if (obj.Has(key) && obj.Get(key).IsBoolean()) {
    return obj.Get(key).As<Napi::Boolean>().Value();
  }
  return defaultVal;
}

inline std::string getStringOption(Napi::Object obj, const std::string& key, const std::string& defaultVal) {
  if (obj.Has(key) && obj.Get(key).IsString()) {
    return obj.Get(key).As<Napi::String>().Utf8Value();
  }
  return defaultVal;
}

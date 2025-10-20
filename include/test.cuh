#ifndef TEST_H
#define TEST_H

#include <functional>

typedef std::vector<std::pair<std::string, std::function<void()>>> FuncVector;

#define TEST(func_name) \
    void func_name(); \
    struct func_name##_registrar { \
        func_name##_registrar() { \
            registered_funcs.push_back({#func_name, func_name}); \
        } \
    } func_name##_instance; \
    void func_name()

#define SKIP(func_name) \
    void func_name()


void test();

#endif
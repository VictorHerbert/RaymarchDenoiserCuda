#ifndef TEST_H
#define TEST_H

#include <vector>
#include <string>
#include <iostream>

enum TestStatus{
    SUCCESS, FAIL, NOT_IMPLEMENTED
};

std::string to_string(TestStatus id);

extern std::vector<std::pair<std::string, TestStatus(*)()>> test_functions;

struct FuncRegistrar {
    FuncRegistrar(std::string str, TestStatus (*ptr)()) {
        test_functions.push_back(std::make_pair(str, ptr));
    }
};

#define TEST(f) \
    TestStatus _test_##f(); \
    static FuncRegistrar _reg_##f(#f, _test_##f); \
    TestStatus _test_##f()

#define SKIP(f) \
    TestStatus f()

void test();

#endif
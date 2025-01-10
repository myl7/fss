// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Yulong Ming (myl7)

#pragma once

#include <stdint.h>

#ifndef kLambda
#define kLambda 16
#endif

void prg(uint8_t *out, const uint8_t *seed);

void group_add(uint8_t *val, const uint8_t *rhs);
void group_neg(uint8_t *val);

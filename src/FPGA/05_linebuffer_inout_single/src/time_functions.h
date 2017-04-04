#pragma once

#include <time.h>

void RecordTime(struct timespec *ts);
double TimeDifference(struct timespec *before, struct timespec *after);

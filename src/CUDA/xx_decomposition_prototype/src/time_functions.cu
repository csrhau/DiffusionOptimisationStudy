#include "time_functions.cuh"

#include <time.h>
#include <sys/time.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

// Use clock_gettime in linux, clock_get_time in OS X.
void RecordTime(struct timespec *ts){
#ifdef __MACH__
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  ts->tv_sec = mts.tv_sec;
  ts->tv_nsec = mts.tv_nsec;
#else
  clock_gettime(CLOCK_MONOTONIC, ts);
#endif
}

double TimeDifference(struct timespec *before, struct timespec *after){
  double deltat_s  = after->tv_sec - before->tv_sec;
  double deltat_ns = after->tv_nsec - before->tv_nsec;
  return deltat_s + deltat_ns*1e-9;
}

#ifndef PROG_INTERFACE_STATIC_LINKH
#define PROG_INTERFACE_STATIC_LINKH


void begin_prog(const char* title,bool lock = false);
void unique_prog(bool unique);
void set_title(const char* title);
bool check_prog(unsigned int now,unsigned int total);
bool prog_aborted(void);
bool is_running(void);
#endif


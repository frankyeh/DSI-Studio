#ifndef PROG_INTERFACE_STATIC_LINKH
#define PROG_INTERFACE_STATIC_LINKH


void begin_prog(const char* title,bool lock = false);
void set_title(const char* title);
int check_prog(int now,unsigned int total);
int prog_aborted(void);
bool is_running(void);
#endif


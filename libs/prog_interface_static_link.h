#ifndef PROG_INTERFACE_STATIC_LINKH
#define PROG_INTERFACE_STATIC_LINKH


extern "C"{
    void begin_prog(const char* title,bool lock = false);
    void end_prog(void);
	void set_title(const char* title);
	void can_cancel(int cancel);
	int check_prog(int now,int total);
	int prog_aborted(void);
}

#endif


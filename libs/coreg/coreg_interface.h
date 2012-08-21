// ---------------------------------------------------------------------------

#ifndef coreg_interfaceH
#define coreg_interfaceH

class LDDMM2 {
private:
	void* handle;

public:
	LDDMM2(unsigned int width, unsigned int height, short* from_image,
		short* to_image);
	~LDDMM2(void);

	unsigned int get_frame_count(void);
	void run(unsigned int frame_number, unsigned int iteration);
	const short* getj(unsigned int index, bool to);
	const short* getv(unsigned int index);

};

class lddmm2_map {
private:
	void* handle;

public:
	lddmm2_map(unsigned int width, unsigned int height);
	~lddmm2_map(void);

	void add_image(const short* from_image);
	void run(unsigned int iteration);
	const short* getj(void);
};

class lddmm3_map {
private:
	void* handle;

public:
	lddmm3_map(unsigned int width, unsigned int height, unsigned int depth);
	~lddmm3_map(void);

	void add_image(const short* from_image);
	void run(unsigned int iteration);
	const short* getj(void);
};

class LDDMM3 {
private:
	void* handle;

public:
	LDDMM3(unsigned int width, unsigned int height, unsigned int depth,
		short* from_image, short* to_image);
	~LDDMM3(void);

	unsigned int get_frame_count(void);
	void run(unsigned int frame_number, unsigned int iteration);
	const short* getj(unsigned int index, bool to);
	const short* getv(unsigned int index);
};
#endif

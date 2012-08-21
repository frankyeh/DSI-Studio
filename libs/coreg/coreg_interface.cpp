// ---------------------------------------------------------------------------
#include "coreg_interface.h"
// ---------------------------------------------------------------------------
extern "C" {
	void* create_lddmm2(unsigned int width, unsigned int height,
		short* from_image, short* to_image);
	unsigned int lddmm2_get_frame_count(void* lddmm);

	void lddmm2_run(void* lddmm, unsigned int frame_number,
		unsigned int iteration);

	const short* lddmm2_getj(void* lddmm, unsigned int index, bool to);
	const short* lddmm2_getv(void* lddmm, unsigned int index);

	void lddmm2_free(void* lddmm);

	// ------------------------------------------------------------
        void*  create_lddmm3(unsigned int width, unsigned int height,
		unsigned int depth, short* from_image, short* to_image);
        unsigned int lddmm3_get_frame_count(void* lddmm);

        void lddmm3_run(void* lddmm, unsigned int frame_number,
		unsigned int iteration);

        const short*  lddmm3_getj(void* lddmm, unsigned int index, bool to);
        const short*  lddmm3_getv(void* lddmm, unsigned int index);

        void lddmm3_free(void* lddmm);
	// ----------------------------------------------------------------

        void*  create_lddmm2_map(unsigned int width, unsigned int height);
        void lddmm2_map_add_image(void* lddmm, const short* image);
        void lddmm2_map_run(void* lddmm, unsigned int iteration);
        const short*  lddmm2_map_getj(void* lddmm);
        void lddmm2_map_free(void* lddmm);

	// ----------------------------------------------------------------

        void*  create_lddmm3_map(unsigned int width, unsigned int height,
		unsigned int depth);
        void lddmm3_map_add_image(void* lddmm, const short* image);
        void lddmm3_map_run(void* lddmm, unsigned int iteration);
        const short*  lddmm3_map_getj(void* lddmm);
        void lddmm3_map_free(void* lddmm);

    }
// ---------------------------------------------------------------------------
LDDMM2::LDDMM2(unsigned int width, unsigned int height, short* from_image,
	short* to_image) {
	handle = create_lddmm2(width, height, from_image, to_image);
}

LDDMM2::~LDDMM2(void) {
	lddmm2_free(handle);
}

unsigned int LDDMM2::get_frame_count(void) {
	return lddmm2_get_frame_count(handle);
}

void LDDMM2::run(unsigned int frame_number, unsigned int iteration) {
	lddmm2_run(handle, frame_number, iteration);
}

const short* LDDMM2::getj(unsigned int index, bool to) {
	return lddmm2_getj(handle, index, to);
}

const short* LDDMM2::getv(unsigned int index) {
	return lddmm2_getv(handle, index);
}

// ---------------------------------------------------------------------------------
LDDMM3::LDDMM3(unsigned int width, unsigned int height, unsigned int depth,
	short* from_image, short* to_image) {
	handle = create_lddmm3(width, height, depth, from_image, to_image);
}

LDDMM3::~LDDMM3(void) {
	lddmm3_free(handle);
}

unsigned int LDDMM3::get_frame_count(void) {
	return lddmm3_get_frame_count(handle);
}

void LDDMM3::run(unsigned int frame_number, unsigned int iteration) {
	lddmm3_run(handle, frame_number, iteration);
}

const short* LDDMM3::getj(unsigned int index, bool to) {
	return lddmm3_getj(handle, index, to);
}

const short* LDDMM3::getv(unsigned int index) {
	return lddmm3_getv(handle, index);
}

// ---------------------------------------------------------------------------------
lddmm2_map::lddmm2_map(unsigned int width, unsigned int height) {
	handle = create_lddmm2_map(width, height);
}

lddmm2_map::~lddmm2_map(void) {
	lddmm2_map_free(handle);
}

void lddmm2_map::add_image(const short* from_image) {
	lddmm2_map_add_image(handle, from_image);
}

void lddmm2_map::run(unsigned int iteration) {
	lddmm2_map_run(handle, iteration);
}

const short* lddmm2_map::getj(void) {
	return lddmm2_map_getj(handle);
}

// ---------------------------------------------------------------------------------
lddmm3_map::lddmm3_map(unsigned int width, unsigned int height,
	unsigned int depth) {
	handle = create_lddmm3_map(width, height, depth);
}

lddmm3_map::~lddmm3_map(void) {
	lddmm3_map_free(handle);
}

void lddmm3_map::add_image(const short* from_image) {
	lddmm3_map_add_image(handle, from_image);
}

void lddmm3_map::run(unsigned int iteration) {
	lddmm3_map_run(handle, iteration);
}

const short* lddmm3_map::getj(void) {
	return lddmm3_map_getj(handle);
}

#ifndef GZIP_INTERFACE_HPP
#define GZIP_INTERFACE_HPP
#include "zlib.h"

#ifdef TIPL_HPP
#include "TIPL/io/gz_stream.hpp"
#else
#include "TIPL/tipl.hpp"
#endif

typedef tipl::io::nifti_base<tipl::io::gz_istream,tipl::io::gz_ostream> gz_nifti;
typedef tipl::io::mat_write_base<tipl::io::gz_ostream> gz_mat_write;
typedef tipl::io::mat_read_base<tipl::io::gz_istream> gz_mat_read;

#endif // GZIP_INTERFACE_HPP

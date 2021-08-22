#ifndef UTIL_EXCHANDLING_H
#define UTIL_EXCHANDLING_H

#include <iostream>


namespace exchandling 
{

	inline void outinfo
	(
		const char* file, 
		const unsigned line,
		const char* str
	)
	{
		std::cout << "ERROR : " << file << "(LINE " << line << ")<" << str << ">" << std::endl;
	}



	inline void empty_data
	(
		const char* file,
		const unsigned line,
		const char* str
	)
	{
		outinfo(file, line, str);
		throw std::runtime_error("MESSAGE : The data is emmpty.");
	}


	inline void mismatch_data_size
	(
		const char* file,
		const unsigned line,
		const char* str
	)
	{
		outinfo(file, line, str);
		throw std::runtime_error("MESSAGE : The size of data and label must be the same.");
	}


	inline void invalid_batch_size
	(
		const char* file,
		const unsigned line,
		const char* str
	)
	{
		outinfo(file, line, str);
		throw std::runtime_error("MESSAGE : The batch is an invalid size.");
	}


	inline void invalid_data_size
	(
		const char* file,
		const unsigned line,
		const char* str
	)
	{
		outinfo(file, line, str);
		throw std::runtime_error("MESSAGE : The data or label is an invalid size.");
	}








} //exchandling






#endif
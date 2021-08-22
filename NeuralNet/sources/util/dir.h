#ifndef UTIL_DIR_H
#define UTIL_DIR_H

#include <string>




inline std::string get_parentdir(const std::string path)
{
	const int str1 = static_cast<int>(path.find_last_of("/")) + 1;
	const int str2 = static_cast<int>(path.find_last_of("\\")) + 1;
	const int i = (str1 > str2) ? str1 : str2;

	if (i < 0) { return ""; }
	return path.substr(0, i);
}


inline std::string get_filename(const std::string path)
{
	const int str1 = static_cast<int>(path.find_last_of("/")) + 1;
	const int str2 = static_cast<int>(path.find_last_of("\\")) + 1;
	const int i = (str1 > str2) ? str1 : str2;

	if (i < 0) { return path; }
	return path.substr(i, path.length());
}








#endif
#ifndef UTIL_CAST_H
#define UTIL_CAST_H

#include <string>
#include <typeinfo>





template <class T>
std::string typename_to_str()
{
	std::string type_info = typeid(T).name();
	size_t i = type_info.find_last_of(" ") + 1;

	if (i < 0) { return ""; }
	return type_info.substr(i, type_info.length());
}



#endif

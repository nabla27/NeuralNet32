/*  LICENSE
	Copyright (c) 2021, nabla All rights reserved.
	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#ifndef UTIL_TIMER_H
#define UTIL_TIMER_H

#include <time.h>
#include <iostream>
#include <conio.h>
#include <string>





//指定した時間の差分(ミリ秒)を経過日時に変換し、標準出力する
void show_etime(const clock_t start, const clock_t end)
{
	const clock_t diff = (end - start) / CLOCKS_PER_SEC;
	const clock_t minutes = diff / 60;
	const clock_t hours = minutes / 60;
	const clock_t days = hours / 24;
	std::cout << "time: "
		<< days << "days "
		<< hours - days * 24 << "hours "
		<< minutes - hours * 60 << "minutes "
		<< diff - minutes * 60 << "seconds "
		<< std::endl;
	(void)_getch();
}


//指定した時間の差分(ミリ秒)を経過日時に変換し、string型として返す
std::string get_string_etime(const clock_t start, const clock_t end)
{
	const clock_t diff = (end - start) / CLOCKS_PER_SEC;
	const clock_t minutes = diff / 60;
	const clock_t hours = minutes / 60;
	const clock_t days = hours / 24;

	std::string output = "";
	output += std::to_string(days) + "days ";
	output += std::to_string(hours - days * 24) + "hours ";
	output += std::to_string(minutes - hours * 60) + "minutes ";
	output += std::to_string(diff - minutes * 60) + "seconds";

	return output;
}



#endif
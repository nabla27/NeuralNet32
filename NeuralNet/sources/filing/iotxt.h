#ifndef FILING_IOTXT_H
#define FILING_IOTXT_H

#include <fstream>
#include <string>



namespace filing {



	class IOtxt {
	private:
		std::string file_name = "./iotxt";
	public:
		IOtxt(std::string f_name, bool rewrite = true);
		IOtxt() {}

		inline void set_path(std::string file_name) { this->file_name = file_name; }

		template <class T1 = const char*, class T2 = const char*, class T3 = const char*, class T4 = const char*, class T5 = const char*, class T6 = const char*>
		void write(T1 val1 = "", T2 val2 = "", T3 val3 = "", T4 val4 = "", T5 val5 = "", T6 val6 = "");
	};


	IOtxt::IOtxt(std::string f_name, bool rew) {
		file_name = f_name;
		if (rew == true) {
			std::ofstream fos(file_name);
		}
	}

	template <class T1, class T2, class T3, class T4, class T5, class T6> void IOtxt::write(T1 val1, T2 val2, T3 val3, T4 val4, T5 val5, T6 val6) {
		std::ofstream fos(file_name, std::ios_base::app);
		fos << val1 << "  " << val2 << "  " << val3 << "  " << val4 << "  " << val5 << "  " << val6 << "\n";
	}

}


#endif //FILING_IOTXT_H

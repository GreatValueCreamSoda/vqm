file(READ "${PTX}" code)
file(WRITE "${HEADER}"
  "#pragma once\nnamespace vqm::cli {\ninline constexpr char kNvdecPackPtx[] = R\"ptx(${code})ptx\";\n}\n")

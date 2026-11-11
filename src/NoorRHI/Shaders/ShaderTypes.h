#pragma once

//! C++ and Slang both include this, so one header can declare the argument structs of a shader with `using namespace NoorRHI;`.
#if defined(__SLANG__)
import NoorRHI;
namespace NoorRHI {}
#else
#include "NoorRHI/Core/Types.h"
#endif

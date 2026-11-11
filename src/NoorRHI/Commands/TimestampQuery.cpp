#include "NoorRHI/Commands/TimestampQuery.h"

#include "NoorRHI/Core/Backend.h"

namespace NoorRHI {

double TimestampQuery::getMilliseconds() const {
    return backend ? backend->getMilliseconds() : 0.0;
}

}

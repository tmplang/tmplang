#include <tmplang/Tree/Source/Exprs.h>

using namespace tmplang;

const source::ExprMatchCaseLhsVal &
source::AggregateDestructurationElem::getValue() const {
  return *Value;
}

SourceLocation source::DataDestructurationElem::getEndLoc() const {
  if (auto comma = getComma()) {
    return comma->getSpan().End;
  }
  return std::visit(
      source::visitors{
          [](const std::unique_ptr<Expr> &val) { return val->getEndLoc(); },
          [](const auto &val) { return val.getEndLoc(); }},
      getValue());
}

SourceLocation source::TupleDestructurationElem::getBeginLoc() const {
  return std::visit(
      source::visitors{
          [](const std::unique_ptr<Expr> &val) { return val->getBeginLoc(); },
          [](const auto &val) { return val.getBeginLoc(); }},
      getValue());
}

SourceLocation source::TupleDestructurationElem::getEndLoc() const {
  if (auto comma = getComma()) {
    return comma->getSpan().End;
  }
  return std::visit(
      source::visitors{
          [](const std::unique_ptr<Expr> &val) { return val->getEndLoc(); },
          [](const auto &val) { return val.getEndLoc(); }},
      getValue());
}

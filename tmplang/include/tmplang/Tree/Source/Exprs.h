#ifndef TMPLANG_TREE_SOURCE_EXPRS_H
#define TMPLANG_TREE_SOURCE_EXPRS_H

#include <tmplang/Tree/Source/Decls.h>
#include <tmplang/Tree/Source/Expr.h>

#include <variant>

namespace tmplang::source {

class ExprIntegerNumber final : public Expr {
public:
  explicit ExprIntegerNumber(SpecificToken<TK_IntegerNumber> num)
      : Expr(Kind::ExprIntegerNumber), Number(num) {}

  const auto &getNumber() const { return Number; }

  tmplang::SourceLocation getBeginLoc() const override {
    return Number.getSpan().Start;
  }
  tmplang::SourceLocation getEndLoc() const override {
    return Number.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprIntegerNumber;
  }

private:
  SpecificToken<TK_IntegerNumber> Number;
};

class TupleElem final : public Node, public TrailingOptComma {
public:
  TupleElem(std::unique_ptr<Expr> val)
      : Node(Kind::TupleElem), Val(std::move(val)) {}

  tmplang::SourceLocation getBeginLoc() const override {
    return Val->getBeginLoc();
  }
  tmplang::SourceLocation getEndLoc() const override {
    return getComma() ? getComma()->getSpan().End : Val->getEndLoc();
  }

  const Expr &getVal() const { return *Val; }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::TupleElem;
  }

private:
  std::unique_ptr<Expr> Val;
};

class ExprTuple final : public Expr {
public:
  ExprTuple(SpecificToken<TK_LParentheses> lParen,
            SmallVector<TupleElem, 4> values,
            SpecificToken<TK_RParentheses> rParen)
      : Expr(Kind::ExprTuple), LParen(lParen), Values(std::move(values)),
        RParen(rParen) {}

  const auto &getLParen() const { return LParen; }
  const auto &getRParen() const { return RParen; }
  ArrayRef<TupleElem> getVals() const { return Values; }

  tmplang::SourceLocation getBeginLoc() const override {
    return LParen.getSpan().Start;
  }
  tmplang::SourceLocation getEndLoc() const override {
    return RParen.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprTuple;
  }

private:
  SpecificToken<TK_LParentheses> LParen;
  SmallVector<TupleElem, 4> Values;
  SpecificToken<TK_RParentheses> RParen;
};

class ExprRet final : public Expr {
public:
  ExprRet(SpecificToken<TK_Ret> ret, std::unique_ptr<Expr> expr = nullptr)
      : Expr(Kind::ExprRet), Ret(ret), ExprToRet(std::move(expr)) {}

  const auto &getRetTk() const { return Ret; }
  const Expr *getReturnedExpr() const { return ExprToRet.get(); }

  tmplang::SourceLocation getBeginLoc() const override {
    return Ret.getSpan().Start;
  }
  tmplang::SourceLocation getEndLoc() const override {
    return ExprToRet ? ExprToRet->getEndLoc() : Ret.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprRet;
  }

private:
  SpecificToken<TK_Ret> Ret;
  std::unique_ptr<Expr> ExprToRet;
};

class ExprVarRef final : public Expr {
public:
  ExprVarRef(SpecificToken<TK_Identifier> id)
      : Expr(Kind::ExprVarRef), Identifier(id) {}

  const auto &getIdentifier() const { return Identifier; }
  llvm::StringRef getName() const { return Identifier.getLexeme(); }

  tmplang::SourceLocation getBeginLoc() const override {
    return Identifier.getSpan().Start;
  }
  tmplang::SourceLocation getEndLoc() const override {
    return Identifier.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprVarRef;
  }

private:
  SpecificToken<TK_Identifier> Identifier;
};

class ExprAggregateDataAccess final : public Expr {
public:
  using BaseNode = std::variant<std::unique_ptr<ExprAggregateDataAccess>,
                                std::unique_ptr<source::ExprTuple>, ExprVarRef>;

  ExprAggregateDataAccess(BaseNode base, SpecificToken<TK_Dot> dot,
                          SpecificToken<TK_Identifier, TK_IntegerNumber> field)
      : Expr(Kind::ExprAggregateDataAccess), Base(std::move(base)), Dot(dot),
        Field(field) {}

  const Expr &getBase() const {
    if (auto *varRef = std::get_if<ExprVarRef>(&Base)) {
      return *varRef;
    }
    if (auto *tupleTy =
            std::get_if<std::unique_ptr<source::ExprTuple>>(&Base)) {
      return **tupleTy;
    }
    return *std::get<std::unique_ptr<ExprAggregateDataAccess>>(Base);
  }

  llvm::StringRef getBaseName() const {
    assert(!std::holds_alternative<std::unique_ptr<source::ExprTuple>>(Base) &&
           "Cannot get base name of tuple");
    if (auto *varRef = std::get_if<ExprVarRef>(&Base)) {
      return varRef->getName();
    }
    return std::get<std::unique_ptr<ExprAggregateDataAccess>>(Base)
        ->getBaseName();
  }

  const auto &getDot() const { return Dot; }
  const auto &getAccessedField() const { return Field; }

  llvm::StringRef getFieldName() const {
    assert(Field.is(TK_Identifier));
    return Field.getLexeme();
  }
  int32_t getNumber() const {
    assert(Field.is(TK_IntegerNumber));
    return Field.getNumber();
  }

  tmplang::SourceLocation getBeginLoc() const override {
    return getBase().getBeginLoc();
  }
  tmplang::SourceLocation getEndLoc() const override {
    return Field.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprAggregateDataAccess;
  }

private:
  BaseNode Base;
  SpecificToken<TK_Dot> Dot;
  SpecificToken<TK_Identifier, TK_IntegerNumber> Field;
};

class VoidPlaceholder : public Node {
public:
  VoidPlaceholder(Token tk) : Node(Kind::VoidPlaceholder), Tk(std::move(tk)) {}

  tmplang::SourceLocation getBeginLoc() const override {
    return Tk.getSpan().Start;
  }

  tmplang::SourceLocation getEndLoc() const override {
    return Tk.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::VoidPlaceholder;
  }

private:
  SpecificToken<TK_Underscore> Tk;
};

class Otherwise : public Node {
public:
  Otherwise(Token tk) : Node(Kind::Otherwise), Tk(std::move(tk)) {}

  tmplang::SourceLocation getBeginLoc() const override {
    return Tk.getSpan().Start;
  }

  tmplang::SourceLocation getEndLoc() const override {
    return Tk.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::Otherwise;
  }

private:
  SpecificToken<TK_Otherwise> Tk;
};

/// Utility to be able to call std::visit over a a std::variant with a lambdas
template <class... Ts> struct visitors : Ts... { using Ts::operator()...; };
template <class... Ts> visitors(Ts...) -> visitors<Ts...>;

class TupleDestructuration;
class DataDestructuration;
class UnionDestructuration;

using ExprMatchCaseLhsVal =
    std::variant<std::unique_ptr<Expr>, PlaceholderDecl, VoidPlaceholder,
                 TupleDestructuration, DataDestructuration,
                 UnionDestructuration>;

class AggregateDestructurationElem : public Node {
public:
  AggregateDestructurationElem(Node::Kind kind,
                               std::unique_ptr<ExprMatchCaseLhsVal> val)
      : Node(kind), Value(std::move(val)) {}

  const ExprMatchCaseLhsVal &getValue() const;

private:
  std::unique_ptr<ExprMatchCaseLhsVal> Value;
};

class TupleDestructurationElem : public AggregateDestructurationElem,
                                 public TrailingOptComma {
public:
  TupleDestructurationElem(std::unique_ptr<ExprMatchCaseLhsVal> value)
      : AggregateDestructurationElem(Kind::TupleDestructurationElem,
                                     std::move(value)) {}

  tmplang::SourceLocation getBeginLoc() const override;
  tmplang::SourceLocation getEndLoc() const override;

  static bool classof(const Node *node) {
    return node->getKind() == Kind::TupleDestructurationElem;
  }
};

class DataDestructurationElem : public AggregateDestructurationElem,
                                public TrailingOptComma {
public:
  DataDestructurationElem(SpecificToken<TK_Identifier> id,
                          SpecificToken<TK_Colon> colon,
                          std::unique_ptr<ExprMatchCaseLhsVal> value)
      : AggregateDestructurationElem(Kind::DataDestructurationElem,
                                     std::move(value)),
        Id(std::move(id)), Colon(std::move(colon)) {}

  const auto &getId() const { return Id; }
  const auto &getColon() const { return Colon; }

  tmplang::SourceLocation getBeginLoc() const override {
    return Id.getSpan().Start;
  }
  tmplang::SourceLocation getEndLoc() const override;

  static bool classof(const Node *node) {
    return node->getKind() == Kind::DataDestructurationElem;
  }

private:
  SpecificToken<TK_Identifier> Id;
  SpecificToken<TK_Colon> Colon;
};

class TupleDestructuration : public Node {
public:
  TupleDestructuration(Token lParen,
                       std::vector<TupleDestructurationElem> tupleElems,
                       Token rParen)
      : Node(Kind::TupleDestructuration), LhsParen(std::move(lParen)),
        TupleElems(std::move(tupleElems)), RhsParen(std::move(rParen)) {}

  const auto &getLhsParen() const { return LhsParen; }
  ArrayRef<TupleDestructurationElem> getTupleElems() const {
    return TupleElems;
  }
  const auto &getRhsParen() const { return RhsParen; }

  tmplang::SourceLocation getBeginLoc() const override {
    return LhsParen.getSpan().Start;
  }

  tmplang::SourceLocation getEndLoc() const override {
    return RhsParen.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::TupleDestructuration;
  }

private:
  SpecificToken<TK_LParentheses> LhsParen;
  std::vector<TupleDestructurationElem> TupleElems;
  SpecificToken<TK_RParentheses> RhsParen;
};

class DataDestructuration : public Node {
public:
  DataDestructuration(Token lbracket,
                      std::vector<DataDestructurationElem> structElems,
                      Token rbracket)
      : Node(Kind::DataDestructuration), LhsBracket(std::move(lbracket)),
        DataElems(std::move(structElems)), RhsBracket(std::move(rbracket)) {}

  const auto &getLhsBracket() const { return LhsBracket; }
  ArrayRef<DataDestructurationElem> getDataElems() const { return DataElems; }
  const auto &getRhsBracket() const { return RhsBracket; }

  tmplang::SourceLocation getBeginLoc() const override {
    return LhsBracket.getSpan().Start;
  }

  tmplang::SourceLocation getEndLoc() const override {
    return RhsBracket.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::DataDestructuration;
  }

public:
  SpecificToken<TK_LKeyBracket> LhsBracket;
  std::vector<DataDestructurationElem> DataElems;
  SpecificToken<TK_RKeyBracket> RhsBracket;
};

class UnionDestructuration : public Node {
public:
  UnionDestructuration(SpecificToken<TK_Identifier> alternative,
                       DataDestructuration dataDes)
      : Node(Kind::UnionDestructuration), Alternative(std::move(alternative)),
        DataDes(std::move(dataDes)) {}

  StringRef getAlternativeStr() const { return Alternative.getLexeme(); }
  const auto &getAlternative() const { return Alternative; }
  const DataDestructuration &getDataDestructuration() const { return DataDes; }

  tmplang::SourceLocation getBeginLoc() const override {
    return Alternative.getSpan().Start;
  }

  tmplang::SourceLocation getEndLoc() const override {
    return DataDes.getEndLoc();
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::UnionDestructuration;
  }

private:
  SpecificToken<TK_Identifier> Alternative;
  DataDestructuration DataDes;
};

class ExprMatchCase final : public Node, public TrailingOptComma {
public:
  using Lhs = std::variant<std::unique_ptr<ExprMatchCaseLhsVal>, Otherwise>;
  using Rhs = std::unique_ptr<Expr>;

  ExprMatchCase(Lhs lhs, SpecificToken<TK_RArrow> arrow, Rhs rhs)
      : Node(Kind::ExprMatchCase), LhsOfCase(std::move(lhs)),
        Arrow(std::move(arrow)), RhsOfCases(std::move(rhs)) {}

  const Lhs &getLhs() const { return LhsOfCase; }
  const auto &getArrow() const { return Arrow; }
  const Rhs &getRhs() const { return RhsOfCases; }

  tmplang::SourceLocation getBeginLoc() const override {
    if (auto *otherwise = std::get_if<Otherwise>(&LhsOfCase)) {
      return otherwise->getBeginLoc();
    }
    return std::visit(
        visitors{
            [](const std::unique_ptr<Expr> &val) { return val->getBeginLoc(); },
            [](const auto &val) { return val.getBeginLoc(); }},
        *std::get<std::unique_ptr<ExprMatchCaseLhsVal>>(LhsOfCase));
  }

  tmplang::SourceLocation getEndLoc() const override {
    return RhsOfCases->getEndLoc();
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprMatchCase;
  }

private:
  Lhs LhsOfCase;
  SpecificToken<TK_RArrow> Arrow;
  Rhs RhsOfCases;
};

class ExprMatch final : public Expr {
public:
  ExprMatch(SpecificToken<TK_Match> match, std::unique_ptr<Expr> expr,
            SpecificToken<TK_LKeyBracket> lkeyBracket,
            SmallVectorImpl<ExprMatchCase> &&cases,
            SpecificToken<TK_RKeyBracket> rkeyBracket)
      : Expr(Kind::ExprMatch), Match(std::move(match)),
        Expression(std::move(expr)), LKeyBracket(std::move(lkeyBracket)),
        Cases(std::move(cases)), RKeyBracket(std::move(rkeyBracket)) {}

  const auto &getMatch() const { return Match; }
  const Expr &getMatchedExpr() const { return *Expression; }
  const auto &getLKeyBracket() const { return LKeyBracket; }
  ArrayRef<ExprMatchCase> getCases() const { return Cases; }
  const auto &getRKeyBracket() const { return RKeyBracket; }

  tmplang::SourceLocation getBeginLoc() const override {
    return Match.getSpan().Start;
  }
  tmplang::SourceLocation getEndLoc() const override {
    return RKeyBracket.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprMatch;
  }

private:
  SpecificToken<TK_Match> Match;
  std::unique_ptr<Expr> Expression;
  SpecificToken<TK_LKeyBracket> LKeyBracket;
  llvm::SmallVector<ExprMatchCase, 4> Cases;
  SpecificToken<TK_RKeyBracket> RKeyBracket;
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_EXPRS_H

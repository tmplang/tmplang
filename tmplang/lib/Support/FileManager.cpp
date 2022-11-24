#include <tmplang/Support/FileManager.h>

#include <llvm/Support/VirtualFileSystem.h>

using namespace tmplang;

FileEntry::~FileEntry() {}

const TargetFileEntry *
FileManager::findOrOpenTargetFile(llvm::StringRef filePath) {
  if (TargetFile) {
    return TargetFile.get();
  }

  auto errorOrFile = FileSystem->openFileForRead(filePath);
  if (!errorOrFile) {
    // TODO: Report error
    return nullptr;
  }

  std::unique_ptr<llvm::vfs::File> file = std::move(errorOrFile.get());

  const auto &status = file->status();
  const auto &name = file->getName();
  if (!status || !name) {
    // TODO: Report Error
    return nullptr;
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      file->getBuffer(*name);
  if (!buffer) {
    // TODO: Report error
    return nullptr;
  }

  assert(!TargetFile && "Only one target file");
  TargetFile = std::make_unique<TargetFileEntry>(
      status->getUniqueID(), std::move(file), std::move(buffer.get()),
      static_cast<off_t>(status->getSize()),
      llvm::sys::toTimeT(status->getLastModificationTime()), *name);

  return TargetFile.get();
}

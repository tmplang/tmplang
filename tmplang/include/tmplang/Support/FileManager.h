#ifndef TMPLANG_SUPPORT_FILEMANAGER_H
#define TMPLANG_SUPPORT_FILEMANAGER_H

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/VirtualFileSystem.h>

namespace tmplang {

class FileEntry {
public:
  enum Kind {
    K_TargetFile // Target file to compile
    // We could add here other types of files, like:
    //   - Module files
    //   - Embeded files (to paste files in compile time)
    //   - etc
  };

  FileEntry(Kind kind, llvm::sys::fs::UniqueID uniqueId,
            std::unique_ptr<llvm::vfs::File> file,
            std::unique_ptr<llvm::MemoryBuffer> content, off_t size,
            time_t modTime)
      : FileEntryKind(kind), UniqueID(uniqueId), File(std::move(file)),
        Content(std::move(content)), Size(size), ModTime(modTime) {}
  virtual ~FileEntry() = 0;

  Kind getKind() const { return FileEntryKind; }

private:
  Kind FileEntryKind;

public:
  llvm::sys::fs::UniqueID UniqueID;
  std::unique_ptr<llvm::vfs::File> File;
  std::unique_ptr<llvm::MemoryBuffer> Content;
  off_t Size;
  time_t ModTime;
};

class TargetFileEntry : public FileEntry {
public:
  TargetFileEntry(llvm::sys::fs::UniqueID uniqueId,
                  std::unique_ptr<llvm::vfs::File> file,
                  std::unique_ptr<llvm::MemoryBuffer> content, off_t size,
                  time_t modTime, std::string realPath)
      : FileEntry(K_TargetFile, uniqueId, std::move(file), std::move(content),
                  size, modTime),
        RealPathName(std::move(realPath)) {}

  static bool classof(const FileEntry *fileEntry) {
    return fileEntry->getKind() == K_TargetFile;
  }

  std::string RealPathName;
};

class FileManager {
public:
  FileManager(std::unique_ptr<llvm::vfs::FileSystem> fs)
      : FileSystem(std::move(fs)) {}

  const TargetFileEntry *findOrOpenTargetFile(llvm::StringRef filePath);

private:
  std::unique_ptr<llvm::vfs::FileSystem> FileSystem;
  std::unique_ptr<TargetFileEntry> TargetFile;
};

} // namespace tmplang

#endif // TMPLANG_SUPPORT_FILEMANAGER_H

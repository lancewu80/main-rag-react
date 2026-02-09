#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// ===== 跨平台UTF-8路徑處理 =====
#ifdef _WIN32
// Windows需要將UTF-8轉為寬字符
static FILE* fopen_utf8(const char *filename, const char *mode) {
    // 計算需要的寬字符數
    int wlen = MultiByteToWideChar(CP_UTF8, 0, filename, -1, NULL, 0);
    if (wlen == 0) return NULL;
    
    wchar_t *wfilename = (wchar_t*)malloc(wlen * sizeof(wchar_t));
    if (!wfilename) return NULL;
    
    MultiByteToWideChar(CP_UTF8, 0, filename, -1, wfilename, wlen);
    
    // 將mode也轉為寬字符
    int wmode_len = MultiByteToWideChar(CP_UTF8, 0, mode, -1, NULL, 0);
    wchar_t *wmode = (wchar_t*)malloc(wmode_len * sizeof(wchar_t));
    if (!wmode) {
        free(wfilename);
        return NULL;
    }
    MultiByteToWideChar(CP_UTF8, 0, mode, -1, wmode, wmode_len);
    
    FILE *fp = _wfopen(wfilename, wmode);
    
    free(wfilename);
    free(wmode);
    return fp;
}
#else
#define fopen_utf8 fopen
#endif

// ===== 詳細的錯誤診斷 =====
EXPORT double fast_write(const char *filename, const char *content) {
    clock_t start, end;
    start = clock();

    printf("\n[C DLL] =====  Write Operation Started =====\n");
    printf("[C DLL] Target file: %s\n", filename);
    printf("[C DLL] Content length: %zu bytes\n", strlen(content));

    // 使用UTF-8安全的fopen
    FILE *fp = fopen_utf8(filename, "wb");
    
    if (fp == NULL) {
        int err = errno;
        printf("[C DLL] ❌ ERROR: Cannot open file\n");
        printf("[C DLL]    Errno: %d (%s)\n", err, strerror(err));
        
#ifdef _WIN32
        // Windows額外診斷
        DWORD win_err = GetLastError();
        printf("[C DLL]    Windows Error Code: %lu\n", win_err);
        
        // 檢查路徑是否存在
        DWORD attrs = GetFileAttributesA(filename);
        if (attrs == INVALID_FILE_ATTRIBUTES) {
            printf("[C DLL]    Path does not exist or is inaccessible\n");
        }
#endif
        return -1.0;
    }

    printf("[C DLL] ✅ File opened successfully\n");

    // 寫入內容
    size_t written = fwrite(content, 1, strlen(content), fp);
    
    if (written != strlen(content)) {
        printf("[C DLL] ❌ ERROR: Partial write (%zu/%zu bytes)\n", 
               written, strlen(content));
        fclose(fp);
        return -2.0;
    }

    printf("[C DLL] ✅ Wrote %zu bytes successfully\n", written);
    
    fclose(fp);
    end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("[C DLL] ⏱️  Operation completed in %.6f seconds\n", elapsed);
    printf("[C DLL] =====================================\n\n");
    
    return elapsed;
}

// ===== 批量寫入優化(適合大檔案) =====
EXPORT double fast_write_buffered(const char *filename, const char *content, size_t buffer_size) {
    if (buffer_size == 0) buffer_size = 65536; // 預設64KB buffer
    
    clock_t start = clock();
    
    FILE *fp = fopen_utf8(filename, "wb");
    if (fp == NULL) return -1.0;
    
    // 設定自訂buffer大小
    char *buffer = (char*)malloc(buffer_size);
    if (buffer) {
        setvbuf(fp, buffer, _IOFBF, buffer_size);
    }
    
    size_t len = strlen(content);
    size_t written = fwrite(content, 1, len, fp);
    
    if (buffer) free(buffer);
    fclose(fp);
    
    if (written != len) return -2.0;
    
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

// ===== 追加寫入功能 =====
EXPORT double fast_append(const char *filename, const char *content) {
    clock_t start = clock();
    
    FILE *fp = fopen_utf8(filename, "ab");
    if (fp == NULL) return -1.0;
    
    if (fputs(content, fp) == EOF) {
        fclose(fp);
        return -2.0;
    }
    
    fclose(fp);
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

// ===== 檔案讀取 =====
EXPORT char* fast_read(const char *filename, size_t *out_size) {
    FILE *fp = fopen_utf8(filename, "rb");
    if (fp == NULL) return NULL;
    
    // 取得檔案大小
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    char *buffer = (char*)malloc(size + 1);
    if (!buffer) {
        fclose(fp);
        return NULL;
    }
    
    size_t read_size = fread(buffer, 1, size, fp);
    buffer[read_size] = '\0';
    
    if (out_size) *out_size = read_size;
    
    fclose(fp);
    return buffer;
}
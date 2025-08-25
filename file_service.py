import os
import shutil
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import structlog
from fastapi import UploadFile, HTTPException

logger = structlog.get_logger()

class FileService:
    """Service for handling file uploads and storage"""
    
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        
        # Supported file types and their max sizes (in MB)
        self.supported_types = {
            '.pdf': 50,    # 50MB
            '.docx': 20,   # 20MB
            '.txt': 10,    # 10MB
            '.csv': 10,    # 10MB
            '.xlsx': 20,   # 20MB
            '.xls': 20,    # 20MB
        }
        
        # Create subdirectories for different file types
        for ext in self.supported_types.keys():
            (self.upload_dir / ext[1:]).mkdir(exist_ok=True)
    
    async def save_uploaded_file(self, file: UploadFile) -> Dict[str, Any]:
        """Save an uploaded file and return file info"""
        try:
            # Validate file type
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in self.supported_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_ext}. Supported types: {list(self.supported_types.keys())}"
                )
            
            # Validate file size
            file_size = 0
            content = b""
            
            # Read file content in chunks to check size
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := await file.read(chunk_size):
                content += chunk
                file_size += len(chunk)
                
                # Check if file exceeds max size
                if file_size > self.supported_types[file_ext] * 1024 * 1024:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File size exceeds maximum allowed size of {self.supported_types[file_ext]}MB"
                    )
            
            # Generate unique filename
            file_id = str(uuid.uuid4())
            safe_filename = f"{file_id}{file_ext}"
            
            # Save file to appropriate subdirectory
            file_path = self.upload_dir / file_ext[1:] / safe_filename
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Get file info
            file_info = {
                "file_id": file_id,
                "original_name": file.filename,
                "saved_name": safe_filename,
                "file_path": str(file_path),
                "file_size": file_size,
                "file_type": file_ext,
                "content_type": file.content_type or "application/octet-stream"
            }
            
            logger.info("File uploaded successfully", 
                       file_id=file_id, 
                       original_name=file.filename,
                       file_size=file_size)
            
            return file_info
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error saving uploaded file", error=str(e), filename=file.filename)
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    def get_file_path(self, file_id: str, file_type: str) -> Optional[str]:
        """Get the file path for a given file ID and type"""
        try:
            file_path = self.upload_dir / file_type[1:] / f"{file_id}{file_type}"
            if file_path.exists():
                return str(file_path)
            return None
        except Exception as e:
            logger.error("Error getting file path", error=str(e), file_id=file_id)
            return None
    
    def delete_file(self, file_id: str, file_type: str) -> bool:
        """Delete a file from storage"""
        try:
            file_path = self.upload_dir / file_type[1:] / f"{file_id}{file_type}"
            if file_path.exists():
                file_path.unlink()
                logger.info("File deleted successfully", file_id=file_id)
                return True
            return False
        except Exception as e:
            logger.error("Error deleting file", error=str(e), file_id=file_id)
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            total_files = 0
            total_size = 0
            files_by_type = {}
            
            for ext in self.supported_types.keys():
                ext_dir = self.upload_dir / ext[1:]
                if ext_dir.exists():
                    files = list(ext_dir.glob("*"))
                    files_by_type[ext] = len(files)
                    total_files += len(files)
                    
                    # Calculate total size for this type
                    for file_path in files:
                        total_size += file_path.stat().st_size
            
            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "files_by_type": files_by_type,
                "supported_types": self.supported_types
            }
            
        except Exception as e:
            logger.error("Error getting storage stats", error=str(e))
            return {}
    
    def cleanup_orphaned_files(self, valid_file_ids: List[str]) -> int:
        """Clean up orphaned files that are no longer referenced"""
        try:
            cleaned_count = 0
            
            for ext in self.supported_types.keys():
                ext_dir = self.upload_dir / ext[1:]
                if ext_dir.exists():
                    for file_path in ext_dir.glob("*"):
                        # Extract file ID from filename
                        file_id = file_path.stem
                        
                        if file_id not in valid_file_ids:
                            try:
                                file_path.unlink()
                                cleaned_count += 1
                                logger.info("Cleaned up orphaned file", file_path=str(file_path))
                            except Exception as e:
                                logger.warning("Failed to delete orphaned file", 
                                             file_path=str(file_path), error=str(e))
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} orphaned files")
            
            return cleaned_count
            
        except Exception as e:
            logger.error("Error cleaning up orphaned files", error=str(e))
            return 0
    
    def get_supported_types(self) -> Dict[str, int]:
        """Get supported file types and their max sizes"""
        return self.supported_types.copy()


"""
File Utilities Module
File system operations and data management utilities
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class FileUtils:
    """File system utilities"""
    
    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if not
        
        Args:
            directory: Directory path
            
        Returns:
            Path object
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def safe_filename(text: str, max_length: int = 50) -> str:
        """
        Create safe filename from text
        
        Args:
            text: Input text
            max_length: Maximum filename length
            
        Returns:
            Safe filename string
        """
        # Remove or replace unsafe characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        filename = ""
        
        for char in text:
            if char in safe_chars:
                filename += char
            elif char in " \t":
                filename += "_"
            # Skip other characters
        
        # Limit length
        if len(filename) > max_length:
            filename = filename[:max_length]
        
        # Ensure not empty
        if not filename:
            filename = "untitled"
        
        return filename
    
    @staticmethod
    def save_json(data: Dict, file_path: Union[str, Path], indent: int = 2):
        """
        Save dictionary as JSON file
        
        Args:
            data: Data to save
            file_path: Output file path
            indent: JSON indentation
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        logger.info(f"Saved JSON: {file_path}")
    
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Dict:
        """
        Load JSON file
        
        Args:
            file_path: JSON file path
            
        Returns:
            Loaded data dictionary
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded JSON: {file_path}")
        return data
    
    @staticmethod
    def copy_file(source: Union[str, Path], 
                  destination: Union[str, Path], 
                  create_dirs: bool = True) -> Path:
        """
        Copy file to destination
        
        Args:
            source: Source file path
            destination: Destination file path
            create_dirs: Create destination directories if needed
            
        Returns:
            Destination path
        """
        source_path = Path(source)
        dest_path = Path(destination)
        
        if create_dirs:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(source_path, dest_path)
        logger.info(f"Copied: {source_path} → {dest_path}")
        
        return dest_path
    
    @staticmethod
    def find_files(directory: Union[str, Path], 
                   pattern: str = "*", 
                   recursive: bool = True) -> List[Path]:
        """
        Find files matching pattern
        
        Args:
            directory: Directory to search
            pattern: File pattern (e.g., "*.wav")
            recursive: Search recursively
            
        Returns:
            List of matching file paths
        """
        directory = Path(directory)
        
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
        
        return files
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """Get file size in bytes"""
        return Path(file_path).stat().st_size
    
    @staticmethod
    def get_directory_size(directory: Union[str, Path]) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        for file_path in Path(directory).rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    @staticmethod
    def cleanup_empty_dirs(directory: Union[str, Path]):
        """Remove empty directories recursively"""
        directory = Path(directory)
        
        for subdir in directory.rglob('*'):
            if subdir.is_dir():
                try:
                    subdir.rmdir()  # Only removes if empty
                    logger.info(f"Removed empty directory: {subdir}")
                except OSError:
                    pass  # Directory not empty
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
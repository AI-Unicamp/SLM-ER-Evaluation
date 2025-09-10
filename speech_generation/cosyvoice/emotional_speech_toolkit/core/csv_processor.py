"""
CSV Processor Module
Handles loading and processing of emotion sentences CSV data
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TextSample:
    """Represents a text sample from CSV with emotion information"""
    text: str
    emotion: str
    csv_column: str
    line_number: int  # ADD LINE NUMBER FIELD
    row_index: int    # ADD ROW INDEX FIELD (0-based pandas index)
    metadata: Optional[Dict] = None

class CSVProcessor:
    """Processes emotion sentences CSV file"""
    
    def __init__(self, csv_file: str):
        """
        Initialize CSV processor
        
        Args:
            csv_file: Path to CSV file containing emotion sentences
        """
        self.csv_file = Path(csv_file)
        self.df = None
        self.text_samples = []
        self.emotion_columns = []
        
        # Load and process CSV
        self._load_csv()
        self._identify_emotion_columns()
        self._extract_text_samples()
    
    def _load_csv(self):
        """Load CSV file"""
        try:
            logger.info(f"Loading CSV file: {self.csv_file}")
            self.df = pd.read_csv(self.csv_file)
            logger.info(f"Loaded {len(self.df)} rows with columns: {list(self.df.columns)}")
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise
    
    def _identify_emotion_columns(self):
        """Identify emotion-related columns in the CSV"""
        # Common emotion patterns
        emotion_patterns = [
            'happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust',
            'joy', 'anger', 'sadness', 'excitement', 'calm', 'stressed'
        ]
        
        emotion_columns = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in emotion_patterns):
                emotion_columns.append(col)
        
        self.emotion_columns = emotion_columns
        logger.info(f"Identified emotion columns: {emotion_columns}")
    
    def _extract_text_samples(self):
        """Extract text samples from all emotion columns"""
        samples = []
        
        for row_idx, row in self.df.iterrows():
            # Calculate line number (1-based, accounting for header)
            line_number = row_idx + 1  # +1 for 0-based index, +1 for header row
            
            for col in self.emotion_columns:
                text = row[col]
                
                # Skip empty/null texts
                if pd.isna(text) or str(text).strip() == '':
                    continue
                
                # Clean text
                text_clean = str(text).strip()
                if len(text_clean) < 3:  # Skip very short texts
                    continue
                
                # Extract emotion from column name
                emotion = self._extract_emotion_from_column(col)
                
                sample = TextSample(
                    text=text_clean,
                    emotion=emotion,
                    csv_column=col,
                    line_number=line_number,  # ADD LINE NUMBER
                    row_index=row_idx,        # ADD ROW INDEX
                    metadata={
                        'source_file': str(self.csv_file),
                        'original_column': col,
                        'pandas_row_index': row_idx
                    }
                )
                samples.append(sample)
        
        self.text_samples = samples
        logger.info(f"Extracted {len(samples)} text samples from {len(self.emotion_columns)} emotion columns")
    
    def _extract_emotion_from_column(self, column_name: str) -> str:
        """Extract emotion name from column name"""
        col_lower = column_name.lower()
        
        # Map common emotion patterns
        emotion_map = {
            'happy': 'happy',
            'joy': 'happy',
            'excitement': 'happy',
            'sad': 'sad',
            'sadness': 'sad',
            'angry': 'angry',
            'anger': 'angry',
            'neutral': 'neutral',
            'calm': 'neutral',
            'surprise': 'surprised',
            'surprised': 'surprised',
            'fear': 'fear',
            'disgust': 'disgust'
        }
        
        for pattern, emotion in emotion_map.items():
            if pattern in col_lower:
                return emotion
        
        # Fallback: use column name directly
        return col_lower.split('_')[0] if '_' in col_lower else col_lower
    
    def get_samples_by_emotion(self, emotion: str) -> List[TextSample]:
        """Get all samples for a specific emotion"""
        return [sample for sample in self.text_samples if sample.emotion.lower() == emotion.lower()]
    
    def get_samples_by_column(self, column: str) -> List[TextSample]:
        """Get all samples from a specific CSV column"""
        return [sample for sample in self.text_samples if sample.csv_column == column]
    
    def get_unique_texts(self) -> List[TextSample]:
        """Get unique texts (removing duplicates)"""
        seen_texts = set()
        unique_samples = []
        
        for sample in self.text_samples:
            text_key = sample.text.lower().strip()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_samples.append(sample)
        
        return unique_samples
    
    def print_summary(self):
        """Print summary of loaded data"""
        print("📊 CSV Dataset Summary")
        print("=" * 40)
        print(f"Source file: {self.csv_file}")
        print(f"Total rows: {len(self.df)}")
        print(f"Total columns: {len(self.df.columns)}")
        print(f"Emotion columns: {len(self.emotion_columns)}")
        print(f"Text samples extracted: {len(self.text_samples)}")
        
        # Emotion distribution
        emotion_counts = {}
        for sample in self.text_samples:
            emotion_counts[sample.emotion] = emotion_counts.get(sample.emotion, 0) + 1
        
        print(f"\n🎭 Emotion distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"   {emotion:12}: {count:4d} samples")
        
        # Column distribution
        print(f"\n📋 Column distribution:")
        for col in self.emotion_columns:
            count = len(self.get_samples_by_column(col))
            print(f"   {col:20}: {count:4d} samples")
        
        # Line number range
        line_numbers = [sample.line_number for sample in self.text_samples]
        if line_numbers:
            print(f"\n📄 Line number range: {min(line_numbers)} - {max(line_numbers)}")
        
        # Sample texts
        print(f"\n📝 Sample texts (first 3):")
        for i, sample in enumerate(self.text_samples[:3]):
            print(f"   Line {sample.line_number}: {sample.text[:60]}...")
    
    def validate_for_synthesis(self) -> Tuple[bool, List[str]]:
        """
        Validate dataset for synthesis pipeline
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if we have samples
        if not self.text_samples:
            issues.append("No text samples found")
        
        # Check emotion coverage
        emotions_found = set(sample.emotion for sample in self.text_samples)
        required_emotions = {'happy', 'sad', 'angry', 'neutral'}
        missing_emotions = required_emotions - emotions_found
        
        if missing_emotions:
            issues.append(f"Missing required emotions: {missing_emotions}")
        
        # Check text quality
        short_texts = [s for s in self.text_samples if len(s.text) < 10]
        if short_texts:
            issues.append(f"Found {len(short_texts)} very short texts (< 10 chars)")
        
        # Check for empty texts
        empty_texts = [s for s in self.text_samples if not s.text.strip()]
        if empty_texts:
            issues.append(f"Found {len(empty_texts)} empty texts")
        
        # Check line number consistency
        line_numbers = [sample.line_number for sample in self.text_samples]
        if len(set(line_numbers)) != len(line_numbers):
            issues.append("Duplicate line numbers detected")
        
        # Check for reasonable line number range
        if line_numbers:
            max_line = max(line_numbers)
            if max_line > len(self.df) + 10:  # Allow some buffer
                issues.append(f"Line numbers seem inconsistent (max: {max_line}, rows: {len(self.df)})")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def export_processed_data(self, output_file: str):
        """Export processed data for inspection"""
        output_path = Path(output_file)
        
        # Create DataFrame from text samples
        export_data = []
        for sample in self.text_samples:
            export_data.append({
                'line_number': sample.line_number,
                'row_index': sample.row_index,
                'emotion': sample.emotion,
                'csv_column': sample.csv_column,
                'text': sample.text,
                'text_length': len(sample.text),
                'source_file': sample.metadata.get('source_file', '')
            })
        
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(output_path, index=False)
        
        logger.info(f"Exported processed data to: {output_path}")
        return str(output_path)
    
    def get_statistics(self) -> Dict:
        """Get detailed statistics about the dataset"""
        stats = {
            'total_samples': len(self.text_samples),
            'total_rows': len(self.df),
            'emotion_columns': len(self.emotion_columns),
            'unique_texts': len(self.get_unique_texts()),
            'emotions': {},
            'columns': {},
            'text_lengths': {},
            'line_number_range': None
        }
        
        # Emotion statistics
        for sample in self.text_samples:
            emotion = sample.emotion
            stats['emotions'][emotion] = stats['emotions'].get(emotion, 0) + 1
        
        # Column statistics
        for sample in self.text_samples:
            col = sample.csv_column
            stats['columns'][col] = stats['columns'].get(col, 0) + 1
        
        # Text length statistics
        lengths = [len(sample.text) for sample in self.text_samples]
        if lengths:
            stats['text_lengths'] = {
                'min': min(lengths),
                'max': max(lengths),
                'avg': sum(lengths) / len(lengths)
            }
        
        # Line number range
        line_numbers = [sample.line_number for sample in self.text_samples]
        if line_numbers:
            stats['line_number_range'] = {
                'min': min(line_numbers),
                'max': max(line_numbers),
                'count': len(set(line_numbers))
            }
        
        return stats
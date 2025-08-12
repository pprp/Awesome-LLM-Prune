#!/usr/bin/env python3
"""
ArXiv Paper Update Script for LLM Pruning Repository
This script searches for new papers related to LLM pruning and updates the README.md
"""

import re
import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import requests
import feedparser
from dateutil.parser import parse as parse_date

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArXivTracker:
    """Handles tracking of processed arXiv papers to avoid duplicates"""
    
    def __init__(self, tracking_file: str = "scripts/processed_papers.json"):
        self.tracking_file = tracking_file
        self.processed_ids: Set[str] = set()
        self.load_processed_ids()
        
    def load_processed_ids(self):
        """Load previously processed arXiv IDs from JSON file"""
        try:
            if os.path.exists(self.tracking_file):
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_ids = set(data.get('processed_arxiv_ids', []))
                    logger.info(f"Loaded {len(self.processed_ids)} previously processed arXiv IDs")
            else:
                logger.info("No tracking file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading tracking file: {e}")
            
    def save_processed_ids(self):
        """Save processed arXiv IDs to JSON file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
            
            # Prepare data structure
            data = {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_papers": len(self.processed_ids),
                "processed_arxiv_ids": sorted(list(self.processed_ids))
            }
            
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Updated tracking file with {len(self.processed_ids)} arXiv IDs")
            
        except Exception as e:
            logger.error(f"Error saving tracking file: {e}")
            
    def is_processed(self, arxiv_id: str) -> bool:
        """Check if an arXiv paper has already been processed"""
        return arxiv_id in self.processed_ids
        
    def add_processed(self, arxiv_id: str):
        """Add an arXiv ID to the processed list"""
        self.processed_ids.add(arxiv_id)
        
    def filter_new_papers(self, papers: List[Dict]) -> List[Dict]:
        """Filter out papers that have already been processed"""
        new_papers = []
        
        for paper in papers:
            if not self.is_processed(paper['id']):
                new_papers.append(paper)
            else:
                logger.info(f"Skipping already processed paper: {paper['title']}")
                
        logger.info(f"Filtered {len(papers)} papers -> {len(new_papers)} new papers")
        return new_papers

class ArXivPaperFetcher:
    """Handles fetching papers from arXiv API"""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self):
        self.search_terms = [
            "pruning",
            "large language model", 
            "LLM",
            "compression",
            "sparsity",
            "structured pruning",
            "unstructured pruning",
            "magnitude pruning",
            "gradual pruning"
        ]
        
    def build_query(self, days_back: int = 1) -> str:
        """Build arXiv API query string - focused on LLM pruning and sparsity"""
        
        # Ensure LLM is ALWAYS included in search results
        # All query parts must include LLM-related terms
        llm_terms = [
            'all:"large language model"',
            'all:"LLM"', 
            'all:"transformer"',
            'all:"language model"'
        ]
        
        pruning_terms = [
            'all:"pruning"',
            'all:"sparse"',
            'all:"sparsity"',
        ]
        
        # Create combinations ensuring LLM + pruning/sparsity
        query_parts = []
        for llm_term in llm_terms:
            for prune_term in pruning_terms:
                query_parts.append(f'({llm_term} AND {prune_term})')
        
        # Add specific LLM pruning method combinations
        specific_combinations = [
            '(all:"BERT" AND all:"pruning")',
            '(all:"GPT" AND all:"pruning")', 
            '(all:"LLaMA" AND all:"pruning")',
            '(all:"transformer" AND all:"magnitude pruning")',
            '(all:"language model" AND all:"structured pruning")',
            '(all:"LLM" AND all:"unstructured pruning")',
            '(all:"transformer" AND all:"lottery ticket")',
            '(all:"language model" AND all:"model compression")'
        ]
        
        query_parts.extend(specific_combinations)
        
        # Join with OR to cast a wider net, but each part ensures LLM + pruning
        query = ' OR '.join(query_parts)
        
        logger.info(f"Built LLM-focused query with {len(query_parts)} combinations")
        logger.info(f"Query: {query[:400]}...")  # Log first 400 chars for debugging
        return query
        
    def is_relevant_paper(self, paper: Dict) -> bool:
        """Check if paper is actually relevant to LLM pruning - LLM is REQUIRED"""
        title_lower = paper['title'].lower()
        abstract_lower = paper['summary'].lower()
        
        # Must have LLM-related terms (REQUIRED - this is the key filter)
        llm_keywords = [
            'large language model', 'language model', 'llm', 'transformer', 
            'bert', 'gpt', 'llama', 'chatgpt', 'generative model', 'nlp',
            'pre-trained model', 'foundation model', 'autoregressive',
            't5', 'opt', 'palm', 'claude', 'gemini', 'qwen', 'baichuan'
        ]
        
        # Must have pruning-related terms 
        prune_keywords = [
            'pruning', 'prune', 'sparse', 'sparsity', 'compression',
            'magnitude pruning', 'structured pruning', 'unstructured pruning',
            'gradual pruning', 'lottery ticket', 'network compression', 'weight pruning',
            'channel pruning', 'layer pruning', 'head pruning', 'token pruning'
        ]
        
        # LLM requirement is STRICT - must have LLM terms
        has_llm = any(keyword in title_lower or keyword in abstract_lower for keyword in llm_keywords)
        has_prune = any(keyword in title_lower or keyword in abstract_lower for keyword in prune_keywords)
        
        # Exclude papers that are clearly not about LLM (even stricter exclusion)
        exclude_keywords = [
            'motion generation', 'video', 'image', 'computer vision', 'cv', 'visual',
            'robotics', 'control', 'mechanical', 'hardware design', 'circuit',
            'convolutional neural network', 'cnn', 'image classification', 'object detection',
            'medical imaging', 'recommendation system', 'graph neural', 'reinforcement learning'
        ]
        
        has_exclude = any(keyword in title_lower or keyword in abstract_lower for keyword in exclude_keywords)
        
        # BOTH LLM and pruning terms are required, no exclusions allowed
        is_relevant = has_llm and has_prune and not has_exclude
        
        if not is_relevant:
            if not has_llm:
                logger.info(f"Filtered out - NO LLM terms: {paper['title']}")
            elif not has_prune:
                logger.info(f"Filtered out - NO pruning terms: {paper['title']}")
            else:
                logger.info(f"Filtered out - excluded domain: {paper['title']}")
            
        return is_relevant
        
    def fetch_recent_papers(self, days_back: int = 1) -> List[Dict]:
        query = self.build_query(days_back)
        
        params = {
            'search_query': query,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending',
            'max_results': 200  # Get many results to find recent ones
        }
        
        try:
            logger.info(f"Making arXiv API request with {days_back} days lookback...")
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            logger.info(f"ArXiv API response status: {response.status_code}")
            logger.info(f"Response content length: {len(response.content)} bytes")
            
            # Parse the Atom feed
            feed = feedparser.parse(response.content)
            
            logger.info(f"Feed parsed. Total entries found: {len(feed.entries)}")
            
            if len(feed.entries) == 0:
                logger.warning("No entries found in arXiv response - may indicate query issues")
                return []
            
            papers = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            logger.info(f"Looking for papers newer than: {cutoff_date}")
            
            date_filtered_count = 0
            relevance_filtered_count = 0
            
            for i, entry in enumerate(feed.entries):
                if i < 3:  # Log first few entries for debugging
                    logger.info(f"Entry {i+1}: {entry.title[:100]}...")
                    logger.info(f"  Published: {entry.published}")
                
                # Parse submission date
                published_date = parse_date(entry.published)
                logger.info(f"  Parsed date: {published_date}")
                
                # Check date filter
                if published_date.replace(tzinfo=None) > cutoff_date:
                    date_filtered_count += 1
                    
                    paper = {
                        'id': entry.id.split('/')[-1],  # Extract arXiv ID
                        'title': entry.title.replace('\n', ' ').strip(),
                        'authors': [author.name for author in entry.authors],
                        'summary': entry.summary.replace('\n', ' ').strip(),
                        'published': published_date,
                        'link': entry.id,
                        'pdf_url': entry.id.replace('/abs/', '/pdf/') + '.pdf'
                    }
                    
                    # Apply relevance filter
                    if self.is_relevant_paper(paper):
                        relevance_filtered_count += 1
                        papers.append(paper)
                        logger.info(f"Added relevant paper: {paper['title'][:100]}...")
                    
            logger.info(f"Date filtering: {date_filtered_count} papers within {days_back} days")
            logger.info(f"Relevance filtering: {relevance_filtered_count} relevant papers")
            logger.info(f"Final result: {len(papers)} papers to process")
            return papers
            
        except Exception as e:
            logger.error(f"Error fetching papers: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

class PaperSummarizer:
    """Handles paper summarization using LLM APIs"""
    
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_base_url = os.getenv('OPENAI_BASE_URL')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
    def summarize_paper(self, paper: Dict) -> Optional[str]:
        """Summarize a paper using available LLM APIs"""
        
        prompt = f"""
        Please provide a concise summary of this research paper about LLM pruning in 2-3 sentences. 
        Focus on the key contributions, methods, and results. End with relevant hashtags.
        
        Title: {paper['title']}
        Authors: {', '.join(paper['authors'])}
        Abstract: {paper['summary'][:1500]}...
        
        Format your response as a single paragraph summary followed by hashtags like #Pruning #Sparse #LLM
        """
        
        # Try OpenAI first
        if self.openai_api_key:
            try:
                summary = self._summarize_with_openai(prompt)
                if summary:
                    return summary
            except Exception as e:
                logger.warning(f"OpenAI API failed: {e}")
                
        # Try Anthropic as fallback
        if self.anthropic_api_key:
            try:
                summary = self._summarize_with_anthropic(prompt)
                if summary:
                    return summary
            except Exception as e:
                logger.warning(f"Anthropic API failed: {e}")
                
        # Return a basic summary if APIs fail
        return self._generate_basic_summary(paper)
        
    def _summarize_with_openai(self, prompt: str) -> Optional[str]:
        """Summarize using OpenAI API"""
        try:
            import openai
            
            # Initialize client with custom base URL if provided
            if self.openai_base_url:
                client = openai.OpenAI(
                    api_key=self.openai_api_key,
                    base_url=self.openai_base_url
                )
            else:
                client = openai.OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[
                    {"role": "system", "content": "You are an expert in machine learning and LLM pruning research."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.4
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None
            
    def _summarize_with_anthropic(self, prompt: str) -> Optional[str]:
        """Summarize using Anthropic API"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return None
            
    def _generate_basic_summary(self, paper: Dict) -> str:
        """Generate a basic summary when APIs are unavailable"""
        return f"This paper presents research on {paper['title'].lower()}. {paper['summary'][:150]}... <br/>#Pruning #LLM"

class ReadmeUpdater:
    """Handles updating the README.md file"""
    
    def __init__(self, readme_path: str = "README.md"):
        self.readme_path = readme_path
        
    def load_readme(self) -> str:
        """Load the current README.md content"""
        try:
            with open(self.readme_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading README: {e}")
            return ""
            
    def save_readme(self, content: str) -> bool:
        """Save the updated README.md content"""
        try:
            with open(self.readme_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error saving README: {e}")
            return False
            
    def format_paper_entry(self, paper: Dict, summary: str) -> str:
        """Format a paper entry for the README list format"""
        # Format authors - limit to first 3 for space
        authors = paper['authors'][:3]
        if len(paper['authors']) > 3:
            authors_str = ', '.join(authors) + ', et al.'
        else:
            authors_str = ', '.join(authors)
            
        # Format date
        date_str = paper['published'].strftime('%Y-%m-%d')
        
        # Create the list entry in the new format
        entry = f"""- {paper['title']}
    - Label: <img src=https://img.shields.io/badge/pruning-turquoise.svg >
    - Author: {authors_str}
    - Link: {paper['link'].replace('/abs/', '/pdf/')} 
    - Code: Not available
    - Pub: Arxiv {paper['published'].year}
    - Summary: {summary}
    - 摘要: {summary}"""
        
        return entry
        
    def update_papers_list(self, content: str, new_papers: List[Dict], summaries: List[str]) -> str:
        """Update the papers list with new entries"""
        if not new_papers:
            logger.info("No new papers to add")
            return content
            
        # Find the end of the taxonomy table (after the label row)
        taxonomy_end = content.find('| <img src=https://img.shields.io/badge/benchmark-purple.svg > |')
        if taxonomy_end == -1:
            logger.error("Could not find taxonomy table in README")
            return content
            
        # Find the next newline after the taxonomy table
        insert_pos = content.find('\n', taxonomy_end)
        if insert_pos == -1:
            logger.error("Could not find insertion point after taxonomy table")
            return content
            
        logger.info(f"Adding {len(new_papers)} new papers to README")
            
        # Insert new papers at the beginning of the list (after taxonomy table)
        new_entries = []
        for paper, summary in zip(new_papers, summaries):
            entry = self.format_paper_entry(paper, summary)
            new_entries.append(entry)
            
        # Add spacing and new entries
        new_content = (
            content[:insert_pos + 1] + 
            '\n\n' + 
            '\n\n'.join(new_entries) + '\n\n' +
            content[insert_pos + 1:]
        )
        
        return new_content

def main():
    """Main execution function"""
    logger.info("Starting daily arXiv paper update")
    
    # Initialize components
    tracker = ArXivTracker()
    fetcher = ArXivPaperFetcher()
    summarizer = PaperSummarizer()
    updater = ReadmeUpdater()
    
    # Test mode - just fetch and show papers without API requirement
    test_mode = len(sys.argv) > 1 and sys.argv[1] == "--test"
    
    if test_mode:
        logger.info("Running in test mode - will fetch papers but not update README")
    else:
        # Check if we have API keys
        if not (summarizer.openai_api_key or summarizer.anthropic_api_key):
            logger.error("No LLM API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
            logger.info("You can run in test mode with: python update_papers.py --test")
            sys.exit(1)
        
    # Fetch recent papers
    logger.info("Fetching recent papers from arXiv...")
    papers = fetcher.fetch_recent_papers(days_back=30)  # Increase to 30 days for better results
    
    if not papers:
        logger.info("No new papers found from arXiv - exiting without changes")
        sys.exit(0)
        
    # Filter out papers we've already processed
    new_papers = tracker.filter_new_papers(papers)
    
    if not new_papers:
        logger.info("No new unprocessed papers found - exiting without changes")
        sys.exit(0)
        
    logger.info(f"Processing {len(new_papers)} new papers (limited to max 5 commits per run)...")
    
    # STRICT LIMIT: Maximum 5 commits per run to avoid spam
    MAX_COMMITS_PER_RUN = 5
    papers_to_process = new_papers[:MAX_COMMITS_PER_RUN]
    
    if len(new_papers) > MAX_COMMITS_PER_RUN:
        logger.info(f"Found {len(new_papers)} papers, but limiting to {MAX_COMMITS_PER_RUN} commits per run")
        logger.info(f"Remaining {len(new_papers) - MAX_COMMITS_PER_RUN} papers will be processed in next run")
    
    if test_mode:
        logger.info("Test mode - showing found papers:")
        for i, paper in enumerate(papers_to_process):  # Show limited papers
            logger.info(f"Paper {i+1}: {paper['title']}")
            logger.info(f"  Authors: {', '.join(paper['authors'][:3])}")
            logger.info(f"  Date: {paper['published']}")
            logger.info(f"  Abstract: {paper['summary'][:200]}...")
            logger.info("---")
        logger.info(f"Found {len(new_papers)} total papers, showing first {len(papers_to_process)}. Exiting test mode.")
        sys.exit(0)
    
    # Process each paper separately and create individual commits
    total_added = 0
    
    for i, paper in enumerate(papers_to_process):
        logger.info(f"Processing paper {i+1}/{len(papers_to_process)} (max 5): {paper['title']}")
        
        # Summarize the paper
        summary = summarizer.summarize_paper(paper)
        
        # Load current README
        readme_content = updater.load_readme()
        if not readme_content:
            logger.error("Failed to load README.md")
            continue
            
        # Update with single paper
        updated_content = updater.update_papers_list(readme_content, [paper], [summary])
        
        # Check if content actually changed
        if len(updated_content) != len(readme_content):
            # Save the updated README
            if updater.save_readme(updated_content):
                logger.info(f"Successfully added paper: {paper['title']}")
                total_added += 1
                
                # Mark paper as processed
                tracker.add_processed(paper['id'])
                
                # Create individual commit for this paper
                os.system(f'git add README.md')
                commit_message = f'Add paper: {paper["title"][:60]}{"..." if len(paper["title"]) > 60 else ""}'
                os.system(f'git commit -m "{commit_message}"')
                
            else:
                logger.error(f"Failed to save README.md for paper: {paper['title']}")
        else:
            logger.info(f"Paper content didn't change README, skipped: {paper['title']}")
            
        # Add delay to avoid rate limiting
        time.sleep(1)
    
    # Save the updated tracking file with all processed papers (even if limited to 5)
    # This ensures we don't reprocess the same papers in future runs
    if total_added > 0:
        tracker.save_processed_ids()
        logger.info(f"Successfully processed {total_added}/{len(papers_to_process)} papers with individual commits")
        
        if len(new_papers) > MAX_COMMITS_PER_RUN:
            logger.info(f"Note: {len(new_papers) - MAX_COMMITS_PER_RUN} papers remain for next run")
            logger.info("These will be automatically processed when the script runs again")
        
        logger.info("Git commits created - workflow should detect changes")
    else:
        logger.info("No new papers were added")
        
        # Even if no papers were added, update the tracking file to mark attempted papers as processed
        # This prevents infinite retry of papers that fail to be added
        tracker.save_processed_ids()
        
    logger.info("Paper update completed successfully")

if __name__ == "__main__":
    main()
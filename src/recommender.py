"""
Music Recommendation Engine
Connects emotions to Spotify songs
"""

import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import numpy as np


# Load environment variables
load_dotenv()


class MusicRecommender: 
    """
    Recommends music based on emotional state
    """
    
    def __init__(self):
        """Initialize Spotify client"""
        client_id = os.getenv('SPOTIFY_CLIENT_ID')
        client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            raise ValueError(
                "Missing Spotify credentials. "
                "Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env file"
            )
        
        # Authenticate
        client_credentials_manager = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
        self.sp = spotipy.Spotify(
            client_credentials_manager=client_credentials_manager
        )
        
    def get_recommendations(self, emotion_features, num_tracks=5):
        """
        Get song recommendations based on emotion features
        
        Args: 
            emotion_features: Dict with valence, energy, danceability, tempo
            num_tracks: Number of songs to recommend
            
        Returns: 
            List of track dicts
        """
        # Search for tracks matching the emotion
        search_query = self._create_search_query(emotion_features)
        
        # Get search results
        results = self.sp.search(
            q=search_query,
            type='track',
            limit=50
        )
        
        if not results['tracks']['items']:
            return []
        
        # Get audio features for tracks
        track_ids = [track['id'] for track in results['tracks']['items']]
        audio_features = self. sp.audio_features(track_ids)
        
        # Score and rank tracks
        scored_tracks = []
        for track, features in zip(results['tracks']['items'], audio_features):
            if features is None:
                continue
                
            score = self._calculate_match_score(emotion_features, features)
            scored_tracks.append({
                'track': track,
                'features': features,
                'score':  score
            })
        
        # Sort by score
        scored_tracks.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top tracks
        recommendations = []
        for item in scored_tracks[:num_tracks]:
            track = item['track']
            recommendations. append({
                'name': track['name'],
                'artists': ', '.join([artist['name'] for artist in track['artists']]),
                'album': track['album']['name'],
                'url': track['external_urls']['spotify'],
                'preview_url': track['preview_url'],
                'image_url': track['album']['images'][0]['url'] if track['album']['images'] else None,
                'score': item['score']
            })
        
        return recommendations
    
    def _create_search_query(self, emotion_features):
        """
        Create Spotify search query based on emotion
        
        Args:
            emotion_features: Dict with audio features
            
        Returns: 
            Search query string
        """
        valence = emotion_features['valence']
        energy = emotion_features['energy']
        
        # Determine mood keywords
        if valence > 0.7 and energy > 0.6:
            mood = "happy upbeat energetic"
        elif valence > 0.6: 
            mood = "cheerful positive"
        elif valence < 0.3 and energy < 0.4:
            mood = "sad melancholic"
        elif valence < 0.4 and energy > 0.7:
            mood = "intense aggressive"
        elif energy < 0.4:
            mood = "calm relaxing"
        else:
            mood = "chill"
        
        # Add genre diversity
        genres = ["pop", "indie", "electronic", "rock", "r&b"]
        
        return f"{mood} year: 2015-2024"
    
    def _calculate_match_score(self, target_features, track_features):
        """
        Calculate how well a track matches target emotional features
        
        Args: 
            target_features:  Desired audio features
            track_features:  Actual track audio features
            
        Returns: 
            Match score (0-1)
        """
        # Feature weights
        weights = {
            'valence': 0.4,
            'energy': 0.3,
            'danceability': 0.2,
            'tempo': 0.1
        }
        
        score = 0.0
        
        # Valence match
        valence_diff = abs(target_features['valence'] - track_features['valence'])
        score += weights['valence'] * (1 - valence_diff)
        
        # Energy match
        energy_diff = abs(target_features['energy'] - track_features['energy'])
        score += weights['energy'] * (1 - energy_diff)
        
        # Danceability match
        dance_diff = abs(target_features['danceability'] - track_features['danceability'])
        score += weights['danceability'] * (1 - dance_diff)
        
        # Tempo match (normalize to 0-1 range)
        target_tempo = target_features['tempo']
        track_tempo = track_features['tempo']
        tempo_diff = abs(target_tempo - track_tempo) / 100  # Normalize
        tempo_diff = min(tempo_diff, 1.0)
        score += weights['tempo'] * (1 - tempo_diff)
        
        return score


def test_recommender():
    """Test music recommender"""
    recommender = MusicRecommender()
    
    # Test with happy emotion
    happy_features = {
        'valence': 0.8,
        'energy': 0.7,
        'danceability':  0.7,
        'tempo': 120
    }
    
    print("Testing with HAPPY emotion features:")
    print(happy_features)
    print("\nRecommendations:")
    
    recommendations = recommender.get_recommendations(happy_features, num_tracks=5)
    
    for i, track in enumerate(recommendations, 1):
        print(f"\n{i}. {track['name']}")
        print(f"   Artist: {track['artists']}")
        print(f"   Album:  {track['album']}")
        print(f"   Match Score: {track['score']:.2f}")
        print(f"   URL: {track['url']}")


if __name__ == "__main__":
    test_recommender()
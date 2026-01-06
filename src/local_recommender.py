"""
Local Music Recommendation Engine
Works without Spotify API - uses a local song database
"""

import random


# Local database of songs with emotional features
LOCAL_SONGS = {
    'happy':  [
        {'name':  'Happy', 'artists': 'Pharrell Williams', 'album': 'Despicable Me 2', 'valence': 0.96, 'energy':  0.82},
        {'name': 'Walking on Sunshine', 'artists': 'Katrina and the Waves', 'album':  'Walking on Sunshine', 'valence': 0.93, 'energy': 0.88},
        {'name': 'Uptown Funk', 'artists': 'Mark Ronson ft. Bruno Mars', 'album': 'Uptown Special', 'valence': 0.87, 'energy':  0.91},
        {'name': "Can't Stop the Feeling", 'artists': 'Justin Timberlake', 'album':  'Trolls', 'valence': 0.89, 'energy': 0.80},
        {'name': 'Good as Hell', 'artists': 'Lizzo', 'album': 'Cuz I Love You', 'valence': 0.85, 'energy':  0.76},
        {'name': 'Shake It Off', 'artists': 'Taylor Swift', 'album': '1989', 'valence': 0.88, 'energy':  0.85},
        {'name': 'I Gotta Feeling', 'artists':  'Black Eyed Peas', 'album': 'The E.N.D.', 'valence': 0.83, 'energy': 0.88},
    ],
    'sad': [
        {'name': 'Someone Like You', 'artists': 'Adele', 'album': '21', 'valence': 0.18, 'energy':  0.34},
        {'name': 'Fix You', 'artists': 'Coldplay', 'album': 'X&Y', 'valence': 0.25, 'energy':  0.40},
        {'name':  'The Night We Met', 'artists': 'Lord Huron', 'album': 'Strange Trails', 'valence': 0.20, 'energy':  0.32},
        {'name': 'Mad World', 'artists': 'Gary Jules', 'album': 'Trading Snakeoil for Wolftickets', 'valence': 0.15, 'energy':  0.25},
        {'name': 'Hurt', 'artists':  'Johnny Cash', 'album': 'American IV', 'valence': 0.12, 'energy':  0.22},
        {'name':  'All By Myself', 'artists': 'Celine Dion', 'album': 'Falling Into You', 'valence': 0.22, 'energy':  0.35},
    ],
    'angry': [
        {'name': 'Break Stuff', 'artists': 'Limp Bizkit', 'album': 'Significant Other', 'valence': 0.35, 'energy':  0.95},
        {'name': 'Killing in the Name', 'artists': 'Rage Against the Machine', 'album': 'Rage Against the Machine', 'valence': 0.30, 'energy':  0.97},
        {'name': 'Bodies', 'artists':  'Drowning Pool', 'album': 'Sinner', 'valence': 0.28, 'energy':  0.98},
        {'name': 'Chop Suey! ', 'artists':  'System of a Down', 'album': 'Toxicity', 'valence': 0.32, 'energy':  0.92},
        {'name': 'Given Up', 'artists':  'Linkin Park', 'album': 'Minutes to Midnight', 'valence': 0.25, 'energy': 0.94},
    ],
    'fearful': [
        {'name': 'Creep', 'artists': 'Radiohead', 'album': 'Pablo Honey', 'valence': 0.30, 'energy':  0.55},
        {'name': 'Breathe Me', 'artists':  'Sia', 'album':  'Colour the Small One', 'valence': 0.28, 'energy':  0.35},
        {'name': 'Mad World', 'artists':  'Gary Jules', 'album': 'Trading Snakeoil for Wolftickets', 'valence': 0.15, 'energy': 0.25},
        {'name': 'Everybody Hurts', 'artists': 'R.E.M. ', 'album':  'Automatic for the People', 'valence': 0.22, 'energy': 0.30},
    ],
    'surprised': [
        {'name': 'Superstition', 'artists': 'Stevie Wonder', 'album': 'Talking Book', 'valence': 0.75, 'energy':  0.70},
        {'name': 'Bohemian Rhapsody', 'artists': 'Queen', 'album': 'A Night at the Opera', 'valence':  0.55, 'energy':  0.65},
        {'name':  'Take On Me', 'artists': 'a-ha', 'album': 'Hunting High and Low', 'valence': 0.72, 'energy':  0.85},
        {'name':  'Mr. Brightside', 'artists': 'The Killers', 'album': 'Hot Fuss', 'valence': 0.68, 'energy':  0.80},
    ],
    'disgusted': [
        {'name': 'Bitter Sweet Symphony', 'artists': 'The Verve', 'album': 'Urban Hymns', 'valence': 0.35, 'energy':  0.55},
        {'name': 'Creep', 'artists': 'Radiohead', 'album': 'Pablo Honey', 'valence': 0.30, 'energy': 0.55},
        {'name': 'Boulevard of Broken Dreams', 'artists': 'Green Day', 'album': 'American Idiot', 'valence': 0.32, 'energy':  0.52},
    ],
    'neutral': [
        {'name': 'Viva la Vida', 'artists': 'Coldplay', 'album': 'Viva la Vida', 'valence': 0.50, 'energy':  0.65},
        {'name':  'Clocks', 'artists': 'Coldplay', 'album': 'A Rush of Blood to the Head', 'valence': 0.48, 'energy':  0.70},
        {'name': 'Somewhere Only We Know', 'artists': 'Keane', 'album': 'Hopes and Fears', 'valence': 0.45, 'energy':  0.55},
        {'name': 'The Scientist', 'artists': 'Coldplay', 'album': 'A Rush of Blood to the Head', 'valence': 0.42, 'energy':  0.45},
        {'name': 'Yellow', 'artists':  'Coldplay', 'album': 'Parachutes', 'valence': 0.52, 'energy':  0.50},
    ]
}


class LocalMusicRecommender:
    """
    Recommends music based on emotional state using local database
    No Spotify API required
    """
    
    def __init__(self):
        """Initialize local recommender"""
        self.songs = LOCAL_SONGS
        print("Local Music Recommender initialized (no Spotify required)")
    
    def get_recommendations(self, emotion_features, num_tracks=5):
        """
        Get song recommendations based on emotion features
        
        Args:
            emotion_features: Dict with valence, energy, danceability, tempo
            num_tracks:  Number of songs to recommend
            
        Returns:
            List of track dicts
        """
        valence = emotion_features.get('valence', 0.5)
        energy = emotion_features.get('energy', 0.5)
        
        # Determine primary emotion category from features
        emotion_category = self._features_to_emotion(valence, energy)
        
        # Get songs for this emotion
        available_songs = self.songs.get(emotion_category, self.songs['neutral'])
        
        # Score songs by how well they match the target features
        scored_songs = []
        for song in available_songs:
            score = self._calculate_match_score(emotion_features, song)
            scored_songs.append((song, score))
        
        # Sort by score
        scored_songs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top tracks with required format
        recommendations = []
        for song, score in scored_songs[:num_tracks]:
            recommendations.append({
                'name': song['name'],
                'artists': song['artists'],
                'album':  song['album'],
                'url': f"https://www.youtube.com/results?search_query={song['name']}+{song['artists']}".replace(' ', '+'),
                'preview_url': None,
                'image_url': None,
                'score': score
            })
        
        return recommendations
    
    def _features_to_emotion(self, valence, energy):
        """Map valence/energy to emotion category"""
        if valence > 0.6 and energy > 0.5:
            return 'happy'
        elif valence < 0.35 and energy < 0.45:
            return 'sad'
        elif valence < 0.4 and energy > 0.7:
            return 'angry'
        elif valence < 0.4 and energy < 0.6:
            return 'fearful'
        elif valence > 0.5 and energy > 0.7:
            return 'surprised'
        elif valence < 0.4: 
            return 'disgusted'
        else:
            return 'neutral'
    
    def _calculate_match_score(self, target_features, song):
        """Calculate how well a song matches target emotion features"""
        valence_diff = abs(target_features.get('valence', 0.5) - song.get('valence', 0.5))
        energy_diff = abs(target_features.get('energy', 0.5) - song.get('energy', 0.5))
        
        # Lower difference = higher score
        score = 1.0 - (valence_diff * 0.5 + energy_diff * 0.5)
        return max(0.0, min(1.0, score))


def test_local_recommender():
    """Test local music recommender"""
    recommender = LocalMusicRecommender()
    
    # Test with happy emotion
    happy_features = {
        'valence': 0.8,
        'energy': 0.7,
        'danceability': 0.7,
        'tempo': 120
    }
    
    print("Testing with HAPPY emotion features:")
    print(happy_features)
    print("\nRecommendations:")
    
    recommendations = recommender.get_recommendations(happy_features, num_tracks=5)
    
    for i, track in enumerate(recommendations, 1):
        print(f"\n{i}.{track['name']}")
        print(f"   Artist: {track['artists']}")
        print(f"   Album: {track['album']}")
        print(f"   Match Score: {track['score']:.2f}")
        print(f"   URL: {track['url']}")


if __name__ == "__main__":
    test_local_recommender()
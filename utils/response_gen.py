def generate_responses(emotion, simplified_text):
    suggestions = {
        'happy': ["That's great to hear!", "Keep smiling!", "Want to share more?"],
        'sad': ["I'm here for you.", "Would you like to talk?", "I'm sorry you're feeling this way."],
        'angry': ["Take a deep breath.", "Let's talk it out.", "Do you want a moment to calm down?"],
        'fearful': ["You're safe here.", "Is something worrying you?", "Let’s figure it out together."],
        'disgusted': ["That sounds unpleasant.", "Want to share what happened?"],
        'surprised': ["Wow, that's unexpected!", "Tell me more!"],
        'neutral': ["I'm listening.", "Go on.", "Anything else you’d like to say?"]
    }
    return suggestions.get(emotion, ["I'm here.", "Let's talk."])
